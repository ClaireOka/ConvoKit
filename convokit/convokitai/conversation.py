from collections import defaultdict
from typing import Dict, List, Optional

from convokit.model import Conversation as BaseConversation
from .aiUtil import AI_META_KEYS, LEGACY_AI_KEYS, as_dict, split_ai_fields
from .assistant import Assistant
from .support import Support


def _timestamp_sort_key(timestamp):
    # support timestamps may be strings while utterance timestamps are ints
    if timestamp is None:
        return (0, 0)
    if isinstance(timestamp, (int, float)):
        return (0, timestamp)
    return (1, str(timestamp))


class Conversation(BaseConversation):
    """
    Represents a discrete subset of utterances in the dataset, connected by a reply-to chain,
    optionally accompanied by AI assistants and their support messages.

    Takes the same arguments as convokit.Conversation, plus:

    :param ai_meta: metadata for ConvoKitAI. Recognized keys:

        - "alias": names speakers are referred to in the conversation, keyed by speaker id
        - "assistants": list of Assistants in this conversation
        - "supports": list of Supports created from all assistants in this conversation

    :ivar ai_meta: metadata for ConvoKitAI. Assigning a dict merges it into the existing ai_meta.
    :ivar alias: the speaker id -> alias mapping
    :ivar assistants: the Assistants in this conversation
    :ivar supports: the Supports in this conversation. If ai_meta has no "supports" entry, these are
        collected from the conversation's assistants.
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
        Convert a convokit.Conversation into a convokitai Conversation in place, moving AI fields out of its
        metadata. Metadata deletion must be unlocked by the caller.
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
        return getattr(self, "_ai_meta", {})

    @ai_meta.setter
    def ai_meta(self, value):
        if isinstance(value, dict):
            merged = {**as_dict(getattr(self, "_ai_meta", {})), **value}
            if "assistants" in merged:
                merged["assistants"] = Assistant.normalize_list(merged["assistants"])
            if "supports" in merged:
                merged["supports"] = Support.normalize_list(merged["supports"])
            self._ai_meta = merged
        else:
            self._ai_meta = {}

    @property
    def alias(self) -> Dict[str, str]:
        return as_dict(self.ai_meta.get("alias"))

    @alias.setter
    def alias(self, value):
        self.ai_meta = {"alias": dict(value or {})}

    @property
    def assistants(self) -> List[Assistant]:
        return list(self.ai_meta.get("assistants", []))

    @assistants.setter
    def assistants(self, value):
        self.ai_meta = {"assistants": value}

    def get_assistant(self, assistant_id: str) -> Assistant:
        """
        Get the Assistant with the specified id. Raises a KeyError if there is no such assistant.
        """
        for assistant in self.assistants:
            if assistant.id == assistant_id:
                return assistant
        raise KeyError(assistant_id)

    @property
    def supports(self) -> List[Support]:
        if "supports" in self.ai_meta:
            return list(self.ai_meta["supports"])
        return [support for assistant in self.assistants for support in assistant.supports]

    @supports.setter
    def supports(self, value):
        self.ai_meta = {"supports": value}

    def get_transcript(self, supports: bool = False) -> str:
        """
        Get a plain-text transcript of the conversation, one utterance per line, ordered by timestamp.
        Speaker ids are replaced by their aliases, if present.

        :param supports: whether to include Supports. Each Support is shown right after the utterance it
            replies to (Supports without a reply_to in this conversation are shown first), labeled with
            the speakers that can see it.
        :return: the transcript as a single string
        """
        alias_map = self.alias
        utterances = list(self.iter_utterances())
        utterances.sort(key=lambda u: (_timestamp_sort_key(u.timestamp), str(u.id)))

        assistants_by_id = {assistant.id: assistant for assistant in self.assistants}
        utt_ids = {utt.id for utt in utterances}
        supports_by_reply_to = defaultdict(list)
        for support in sorted(
            self.supports, key=lambda s: (_timestamp_sort_key(s.timestamp), str(s.id))
        ):
            supports_by_reply_to[support.reply_to if support.reply_to in utt_ids else None].append(
                support
            )

        lines = []

        def add_supports(reply_to):
            for support in supports_by_reply_to.get(reply_to, []):
                assistant = assistants_by_id.get(support.assistant_id)
                label = support.assistant_id or "assistant"
                viewers = [alias_map.get(s, s) for s in (assistant.speakers if assistant else [])]
                lines.append(
                    f"    Assistant[{label}]"
                    + (f" -> {', '.join(viewers)}" if viewers else "")
                    + ":"
                )
                if support.draft:
                    lines.append(f"        draft: {support.draft}")
                lines.append(f"        {label}: {support.text}")

        if supports:
            add_supports(None)
        for utterance in utterances:
            speaker_alias = alias_map.get(utterance.speaker.id, utterance.speaker.id)
            lines.append(f"{speaker_alias}: {utterance.text}")
            if supports:
                add_supports(utterance.id)

        return "\n".join(lines)

    def __str__(self):
        return "Conversation('id': {}, 'utterances': {}, 'meta': {}, 'ai_meta': {})".format(
            repr(self.id), self._utterance_ids, self.meta, self.ai_meta
        )
