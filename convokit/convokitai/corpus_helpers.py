"""
Contains functions that help with converting / loading / dumping the AI fields of a ConvoKit AI Corpus.

The base convokit.Corpus machinery constructs plain convokit.Speaker / Utterance / Conversation
objects; the functions here convert those into their convokitai counterparts and move the AI
fields between metadata (on disk) and attributes (in memory).
"""

import json
import os
from typing import Dict, List, Optional

from .aiUtil import AI_META_KEYS, LEGACY_AI_KEYS, LEGACY_CORPUS_KEYS, as_dict, normalize_for_dump
from .assistant import Assistant
from .conversation import Conversation
from .speaker import Speaker
from .support import Support
from .utterance import Utterance

# corpus-level supports file written by earlier iterations of the format
LEGACY_SUPPORTS_FILENAME = "supports.json"

COMPONENT_CLASSES = {"speaker": Speaker, "utterance": Utterance, "conversation": Conversation}


def _unlocked_meta_deletion(corpus, obj_type, fn):
    corpus.meta_index.lock_metadata_deletion[obj_type] = False
    try:
        return fn()
    finally:
        corpus.meta_index.lock_metadata_deletion[obj_type] = True


def upgrade_components(corpus) -> None:
    """
    Convert every convokit.Speaker / Utterance / Conversation in the corpus into its convokitai
    counterpart (in place), moving AI fields out of their metadata. Components that are already
    convokitai objects are left untouched, so this is safe to call repeatedly.
    """
    if not hasattr(corpus, "utterances"):
        # empty corpus
        return

    def objs_of_type(obj_type):
        if obj_type == "speaker":
            # include speakers only reachable through an utterance
            speakers = {id(s): s for s in corpus.iter_speakers()}
            speakers.update({id(u.speaker): u.speaker for u in corpus.iter_utterances()})
            return speakers.values()
        if obj_type == "conversation" and not hasattr(corpus, "conversations"):
            return []
        return list(corpus.iter_objs(obj_type))

    for obj_type, cls in COMPONENT_CLASSES.items():

        def convert():
            converted = False
            for obj in objs_of_type(obj_type):
                if not isinstance(obj, cls):
                    cls._from_base(obj)
                    converted = True
            return converted

        if _unlocked_meta_deletion(corpus, obj_type, convert):
            for key in set(LEGACY_AI_KEYS) | set(AI_META_KEYS[obj_type]):
                corpus.meta_index.del_from_index(obj_type, key)


def extract_corpus_ai_fields(corpus, filename: Optional[str] = None) -> Dict:
    """
    Remove the corpus-level AI fields from corpus.meta and return them, along with any corpus-level
    assistants / supports stored by earlier iterations of the format (including supports.json in the
    corpus directory `filename`), so they can be migrated onto conversations.

    :return: dict with keys "ai_meta", "has_ai" (None if not stored), "legacy_assistants", "legacy_supports"
    """
    ai_meta = as_dict(corpus.meta.get("ai_meta"))
    fields = {
        "ai_meta": {k: v for k, v in ai_meta.items() if k != "assistants"},
        "has_ai": corpus.meta.get("has_ai"),
        "legacy_assistants": ai_meta.get("assistants") or corpus.meta.get("assistants") or [],
        "legacy_supports": corpus.meta.get("supports", []) or [],
    }

    if filename is not None and os.path.isdir(filename):
        support_path = os.path.join(filename, LEGACY_SUPPORTS_FILENAME)
        if os.path.exists(support_path):
            with open(support_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            fields["legacy_supports"] = (
                payload.get("supports", []) if isinstance(payload, dict) else payload
            )

    for key in set(LEGACY_AI_KEYS) | set(AI_META_KEYS["corpus"]) | set(LEGACY_CORPUS_KEYS):
        if key in corpus.meta:
            del corpus.meta[key]

    return fields


def migrate_legacy_assistants_and_supports(corpus, assistants, supports: List[Dict]) -> None:
    """
    Move corpus-level assistants and supports (from earlier iterations of the format) onto the
    Conversations they belong to: each assistant is added to the Conversation matching its
    conversation_id, and each support to the Assistant matching its assistant_id. Supports with
    no matching assistant are added directly to the Conversation matching their conversation_id.
    """
    assistants = Assistant.normalize_list(assistants)
    if not assistants and not supports:
        return

    assistants_by_id = {assistant.id: assistant for assistant in assistants}
    for record in supports or []:
        support = Support.normalize_list([record])
        if not support:
            continue
        support = support[0]
        assistant = assistants_by_id.get(support.assistant_id)
        if assistant is not None:
            if support.id not in {s.id for s in assistant.supports}:
                assistant.supports.append(support)
            continue
        convo_id = record.get("conversation_id") if isinstance(record, dict) else None
        if convo_id is not None and corpus.has_conversation(convo_id):
            convo = corpus.get_conversation(convo_id)
            if support.id not in {s.id for s in convo.supports}:
                convo.supports = convo.supports + [support]

    for assistant in assistants:
        if assistant.conversation_id is not None and corpus.has_conversation(
            assistant.conversation_id
        ):
            convo = corpus.get_conversation(assistant.conversation_id)
            if assistant.id not in {a.id for a in convo.assistants}:
                convo.assistants = convo.assistants + [assistant]


def stash_ai_fields_in_meta(corpus) -> None:
    """
    Write AI fields into the metadata of every component and the corpus (the on-disk format), so the
    base convokit dump writes them out. Undo with unstash_ai_fields_from_meta().
    """
    type_check = corpus.meta_index.type_check
    # keys missing from the index are silently skipped by the base dump, so make sure they get indexed
    corpus.meta_index.enable_type_check()
    try:
        for speaker in corpus.iter_speakers():
            speaker.meta["ai_meta"] = normalize_for_dump(speaker.ai_meta)
            speaker.meta["is_ai"] = speaker.is_ai

        for utt in corpus.iter_utterances():
            utt.meta["ai_meta"] = normalize_for_dump(utt.ai_meta)

        for convo in corpus.iter_conversations():
            convo.meta["ai_meta"] = normalize_for_dump(convo.ai_meta)

        corpus.meta["ai_meta"] = normalize_for_dump(corpus.ai_meta)
        corpus.meta["has_ai"] = corpus.has_ai
    finally:
        if not type_check:
            corpus.meta_index.disable_type_check()


def unstash_ai_fields_from_meta(corpus) -> None:
    """
    Remove the AI fields written by stash_ai_fields_in_meta() from all metadata and from the index.
    """
    for obj_type in COMPONENT_CLASSES:

        def remove():
            for obj in corpus.iter_objs(obj_type):
                for key in AI_META_KEYS[obj_type]:
                    if key in obj.meta:
                        del obj.meta[key]

        _unlocked_meta_deletion(corpus, obj_type, remove)
        for key in AI_META_KEYS[obj_type]:
            corpus.meta_index.del_from_index(obj_type, key)

    for key in AI_META_KEYS["corpus"]:
        if key in corpus.meta:
            del corpus.meta[key]
