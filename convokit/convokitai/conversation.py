from typing import Dict, List, Optional

from convokit.model import Conversation as BaseConversation
from .aiUtil import AI_META_KEYS, LEGACY_AI_KEYS, as_dict, split_ai_fields
from .support import Support


class Conversation(BaseConversation):
    """
    Represents a discrete subset of utterances in the dataset, connected by a reply-to chain,
    optionally accompanied by AI support messages.

    Takes the same arguments as convokit.Conversation, plus:

    :param ai_meta: dictionary of AI-specific attributes of the conversation. Recognized keys are
        "supports" (list of Supports), "assistants" (list of assistant ids), and "alias"
        (mapping of speaker id -> display name used by get_transcript()).

    :ivar ai_meta: AI-specific attributes of the conversation. Assigning a dict merges it into the existing ai_meta.
    :ivar supports: the Supports attached to this conversation
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
            if "supports" in merged:
                merged["supports"] = Support.normalize_list(merged["supports"])
            self._ai_meta = merged
        else:
            self._ai_meta = {}

    @property
    def supports(self) -> List[Support]:
        return Support.normalize_list(self.ai_meta.get("supports", []))

    @supports.setter
    def supports(self, value):
        self.ai_meta = {"supports": Support.normalize_list(value)}

    def get_transcript(self, supports: bool = False) -> str:
        """
        Get a plain-text transcript of the conversation, one utterance per line, ordered by timestamp.
        Speaker ids are replaced by the names in ai_meta["alias"], if present.

        :param supports: whether to interleave each speaker's Supports before the utterance they preceded
        :return: the transcript as a single string
        """
        alias_map = self.ai_meta.get("alias", {}) or {}
        utterances = list(self.iter_utterances())
        try:
            utterances = sorted(
                utterances,
                key=lambda u: (
                    u.timestamp if isinstance(u.timestamp, (int, float)) else 0,
                    str(getattr(u, "id", "")),
                ),
            )
        except Exception:
            pass

        support_records = sorted(
            self.supports,
            key=lambda s: (
                s.timestamp if isinstance(s.timestamp, (int, float)) else 0,
                str(getattr(s, "id", "")),
            ),
        )

        lines = []
        seen_support_ids = set()
        prior_text_by_speaker = {}

        for utterance in utterances:
            speaker_alias = alias_map.get(utterance.speaker.id, utterance.speaker.id)
            prior_texts = prior_text_by_speaker.setdefault(utterance.speaker.id, [])

            if supports:
                for support in support_records:
                    if support.id in seen_support_ids:
                        continue
                    if getattr(support, "speaker_id", None) != utterance.speaker.id:
                        continue
                    if (
                        getattr(support, "timestamp", None) is not None
                        and support.timestamp > utterance.timestamp
                    ):
                        continue

                    reply_to = getattr(support, "reply_to", None) or ""
                    assistant_label = getattr(support, "assistant_id", "assistant")

                    if not reply_to:
                        lines.append(f"    Assistant[{assistant_label}]: {support.text}")
                        seen_support_ids.add(support.id)
                        continue

                    matching_reply = any(
                        existing == reply_to
                        or existing.startswith(reply_to)
                        or reply_to.startswith(existing)
                        for existing in prior_texts
                    )
                    if not matching_reply:
                        continue

                    lines.append(f"    Assistant[{assistant_label}] -> {speaker_alias}:")
                    lines.append(f"        {speaker_alias}: {reply_to}")
                    lines.append(f"        {assistant_label}: {support.text}")
                    seen_support_ids.add(support.id)

            lines.append(f"{speaker_alias}: {utterance.text}")
            prior_texts.append(utterance.text)

        return "\n".join(lines)

    def __str__(self):
        return "Conversation('id': {}, 'utterances': {}, 'meta': {}, 'ai_meta': {})".format(
            repr(self.id), self._utterance_ids, self.meta, self.ai_meta
        )
