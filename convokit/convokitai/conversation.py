"""
The ConvoKit AI Conversation class, with private assistants and their support messages.
"""

from collections import defaultdict
from typing import Dict, List, Optional

from convokit.model import Conversation as BaseConversation
from .aiUtil import AI_META_KEYS, LEGACY_AI_KEYS, as_dict, split_ai_fields, transcript_sort_key
from .private_assistant import PrivateAssistant
from .support import Support


class Conversation(BaseConversation):
    """
    Represents a discrete subset of utterances in the dataset, connected by a reply-to chain,
    optionally accompanied by AI private assistants and their support messages.

    Takes the same arguments as ``convokit.Conversation``, plus:

    :param ai_meta: metadata for ConvoKit AI. Recognized keys:

        - ``"alias"``: names speakers are referred to by in the conversation, keyed by speaker id
        - ``"private_assistants"``: list of PrivateAssistants in this conversation
        - ``"supports"``: list of Supports from all private assistants in this conversation

        If ``meta`` contains an ``"ai_meta"`` entry (as in a dumped corpus), it is moved out of
        ``meta`` and merged with this argument (this argument takes precedence).

    :ivar ai_meta: metadata for ConvoKit AI. Assigning a dict merges it into the existing
        ``ai_meta``.
    :ivar alias: the speaker id -> alias mapping
    :ivar private_assistants: the PrivateAssistants in this conversation
    :ivar supports: all Supports in this conversation: those in ``ai_meta["supports"]``, those of
        its private assistants, and those attached to its utterances.
    """

    def __init__(
        self,
        owner,
        id: Optional[str] = None,
        utterances: Optional[List[str]] = None,
        meta: Optional[Dict] = None,
        ai_meta: Optional[Dict] = None,
    ):
        meta, ai_fields = split_ai_fields(meta, "conversation")
        super().__init__(owner=owner, id=id, utterances=utterances, meta=meta)
        self._ai_meta = {}
        stored_ai_meta = as_dict(ai_fields.get("ai_meta"))
        self.ai_meta = {**stored_ai_meta, **(ai_meta or {})}

    @classmethod
    def _from_base(cls, convo: BaseConversation) -> "Conversation":
        """
        Convert a ``convokit.Conversation`` into a convokitai Conversation in place, moving AI
        fields out of its metadata. Metadata deletion must be unlocked by the caller.
        """
        convo.__class__ = cls
        convo._ai_meta = {}
        convo.ai_meta = as_dict(convo.meta.get("ai_meta"))
        for key in set(LEGACY_AI_KEYS) | set(AI_META_KEYS["conversation"]):
            if key in convo.meta:
                del convo.meta[key]
        return convo

    @property
    def ai_meta(self) -> Dict:
        """The ConvoKit AI metadata of this conversation."""
        return getattr(self, "_ai_meta", {})

    @ai_meta.setter
    def ai_meta(self, value):
        if isinstance(value, dict):
            merged = {**as_dict(getattr(self, "_ai_meta", {})), **value}
            # legacy key: older versions of the format stored private assistants under "assistants"
            if "assistants" in merged:
                legacy = merged.pop("assistants")
                merged.setdefault("private_assistants", legacy)
            if "private_assistants" in merged:
                merged["private_assistants"] = PrivateAssistant.normalize_list(
                    merged["private_assistants"]
                )
            if "supports" in merged:
                merged["supports"] = Support.normalize_list(merged["supports"])
            self._ai_meta = merged
        else:
            self._ai_meta = {}

    @property
    def alias(self) -> Dict[str, str]:
        """Mapping from speaker id to the alias the speaker is referred to by (a copy)."""
        return as_dict(self.ai_meta.get("alias"))

    @alias.setter
    def alias(self, value):
        self.ai_meta = {"alias": dict(value or {})}

    @property
    def private_assistants(self) -> List[PrivateAssistant]:
        """The PrivateAssistants in this conversation (a new list; assign to modify)."""
        return list(self.ai_meta.get("private_assistants", []))

    @private_assistants.setter
    def private_assistants(self, value):
        self.ai_meta = {"private_assistants": value}

    def get_private_assistant(self, assistant_id: str) -> PrivateAssistant:
        """
        Get the PrivateAssistant with the specified id.

        :param assistant_id: id of the private assistant
        :return: the PrivateAssistant
        :raises KeyError: if there is no private assistant with that id in this conversation
        """
        for assistant in self.private_assistants:
            if assistant.id == assistant_id:
                return assistant
        raise KeyError(assistant_id)

    def get_ai_speakers(self) -> List[str]:
        """
        Get the ids of the speakers in this conversation that are AI (``is_ai`` is True).

        :return: a list of speaker ids (empty if there are no AI speakers)
        """
        return [speaker.id for speaker in self.iter_speakers() if getattr(speaker, "is_ai", False)]

    @property
    def supports(self) -> List[Support]:
        """
        All Supports in this conversation (a new list): those in ``ai_meta["supports"]``, those of
        its private assistants, and those attached to its utterances, each Support once (by id).
        Assigning sets ``ai_meta["supports"]``.
        """
        sources = [
            self.ai_meta.get("supports", []),
            *(assistant.supports for assistant in self.private_assistants),
            *(utt.supports for utt in self.iter_utterances()),
        ]
        supports, seen = [], set()
        for source in sources:
            for support in source:
                if support.id not in seen:
                    seen.add(support.id)
                    supports.append(support)
        return supports

    @supports.setter
    def supports(self, value):
        self.ai_meta = {"supports": value}

    def __transcript_entries(self, supports: bool) -> List:
        """
        Return the Utterances ordered by timestamp, with (if ``supports`` is True) each Support
        right after the utterance it replies to. Supports without a ``reply_to`` in this
        conversation come first.
        """
        utterances = sorted(self.iter_utterances(), key=transcript_sort_key)
        if not supports:
            return utterances

        utt_ids = {utt.id for utt in utterances}
        supports_by_reply_to = defaultdict(list)
        for support in sorted(self.supports, key=transcript_sort_key):
            supports_by_reply_to[support.reply_to if support.reply_to in utt_ids else None].append(
                support
            )

        entries = list(supports_by_reply_to.get(None, []))
        for utterance in utterances:
            entries.append(utterance)
            entries.extend(supports_by_reply_to.get(utterance.id, []))
        return entries

    def get_transcript(self, supports: bool = False, until=None) -> str:
        """
        Get a plain-text transcript of the conversation, one utterance per line, ordered by
        timestamp. Speaker ids are replaced by their aliases, if present.

        :param supports: whether to include Supports. Each Support is shown right after the
            utterance it replies to (Supports without a ``reply_to`` in this conversation are shown
            first), labeled with the speakers that can see it.
        :param until: an Utterance or Support of this conversation; if given, the transcript ends
            with it (inclusive). A Support given here is always shown, even if ``supports`` is
            False.
        :return: the transcript as a single string
        :raises ValueError: if ``until`` is not in this conversation
        """
        entries = self.__transcript_entries(supports or isinstance(until, Support))
        if isinstance(until, Support) and not supports:
            entries = [e for e in entries if not isinstance(e, Support) or e.id == until.id]
        if until is not None:
            end = next(
                (
                    i
                    for i, entry in enumerate(entries)
                    if isinstance(entry, Support) == isinstance(until, Support)
                    and entry.id == until.id
                ),
                None,
            )
            if end is None:
                raise ValueError(
                    "{} {!r} is not in conversation {!r}".format(
                        type(until).__name__, until.id, self.id
                    )
                )
            entries = entries[: end + 1]

        alias_map = self.alias
        assistants_by_id = {assistant.id: assistant for assistant in self.private_assistants}
        lines = []
        for entry in entries:
            if isinstance(entry, Support):
                assistant = assistants_by_id.get(entry.private_assistant_id)
                label = entry.private_assistant_id or "private assistant"
                viewers = [alias_map.get(s, s) for s in (assistant.speakers if assistant else [])]
                lines.append(
                    f"    PrivateAssistant[{label}]"
                    + (f" -> {', '.join(viewers)}" if viewers else "")
                    + ":"
                )
                if entry.draft:
                    lines.append(f"        draft: {entry.draft}")
                lines.append(f"        {label}: {entry.text}")
            else:
                speaker_alias = alias_map.get(entry.speaker.id, entry.speaker.id)
                lines.append(f"{speaker_alias}: {entry.text}")

        return "\n".join(lines)

    def __str__(self):
        return "Conversation('id': {}, 'utterances': {}, 'meta': {}, 'ai_meta': {})".format(
            repr(self.id), self._utterance_ids, self.meta, self.ai_meta
        )
