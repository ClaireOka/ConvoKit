"""
Contains functions that help with converting / loading / dumping the AI fields of a ConvoKit AI Corpus.

The base convokit.Corpus machinery constructs plain convokit.Speaker / Utterance / Conversation
objects; the functions here convert those into their convokitai counterparts and move the AI
fields between metadata (on disk) and attributes (in memory).
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Optional

from .aiUtil import AI_META_KEYS, LEGACY_AI_KEYS, as_dict, normalize_for_dump
from .conversation import Conversation
from .speaker import Speaker
from .support import Support
from .utterance import Utterance

SUPPORTS_FILENAME = "supports.json"

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
    Remove the corpus-level AI fields from corpus.meta and return them. If `filename` is a corpus
    directory containing supports.json, supports are read from there instead of corpus.meta.

    :return: dict with keys "ai_meta", "has_ai", "supports", "assistants"
    """
    ai_meta = as_dict(corpus.meta.get("ai_meta"))
    fields = {
        "ai_meta": ai_meta,
        "has_ai": bool(corpus.meta.get("has_ai", False)),
        "supports": corpus.meta.get("supports", []) or [],
        "assistants": ai_meta.get("assistants") or corpus.meta.get("assistants") or {},
    }

    if filename is not None and os.path.isdir(filename):
        support_path = os.path.join(filename, SUPPORTS_FILENAME)
        if os.path.exists(support_path):
            with open(support_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            fields["supports"] = (
                payload.get("supports", []) if isinstance(payload, dict) else payload
            )

    for key in set(LEGACY_AI_KEYS) | set(AI_META_KEYS["corpus"]):
        if key in corpus.meta:
            del corpus.meta[key]

    return fields


def distribute_supports_to_conversations(corpus, supports: List[Support]) -> None:
    """
    Set each Conversation's supports to the corpus-level supports with a matching conversation_id.
    Conversations with no matching corpus-level supports keep the supports they already have.
    """
    by_conversation = defaultdict(list)
    for support in supports:
        if getattr(support, "conversation_id", None) is not None:
            by_conversation[support.conversation_id].append(support)
    for convo_id, records in by_conversation.items():
        if corpus.has_conversation(convo_id):
            corpus.get_conversation(convo_id).supports = records


def stash_ai_fields_in_meta(corpus) -> None:
    """
    Write AI fields into the metadata of every component and the corpus (the on-disk format), so the
    base convokit dump writes them out. Undo with unstash_ai_fields_from_meta().
    """
    supports = corpus.supports
    assistants = corpus.assistants
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
            convo_ai_meta = dict(convo.ai_meta)
            assistant_ids = [
                assistant.id
                for assistant in assistants.values()
                if getattr(assistant, "conversation_id", None) == convo.id
            ]
            relevant = [
                support
                for support in supports
                if getattr(support, "conversation_id", None) == convo.id
            ]
            if assistant_ids:
                convo_ai_meta["assistants"] = assistant_ids
            if relevant:
                convo_ai_meta["supports"] = relevant
            convo.meta["ai_meta"] = normalize_for_dump(convo_ai_meta)

        corpus.meta["ai_meta"] = normalize_for_dump({**corpus.ai_meta, "assistants": assistants})
        corpus.meta["has_ai"] = corpus.has_ai
        corpus.meta["supports"] = normalize_for_dump(supports)
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


def get_dump_dirpath(
    corpus, name: str, base_path: Optional[str], overwrite_existing_corpus: bool
) -> str:
    """
    Get the directory convokit.Corpus.dump() writes to for the given arguments.
    """
    if overwrite_existing_corpus:
        return corpus.corpus_dirpath
    if base_path is None:
        base_path = os.path.join(os.path.expanduser("~/.convokit/"), "saved-corpora/")
    return os.path.join(base_path, name)


def dump_supports(supports: List[Support], dir_name: str) -> None:
    with open(os.path.join(dir_name, SUPPORTS_FILENAME), "w", encoding="utf-8") as f:
        json.dump(
            {"supports": normalize_for_dump(supports)}, f, ensure_ascii=False, indent=2
        )
