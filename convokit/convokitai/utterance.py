from typing import Dict, Optional

from convokit.model import Utterance as BaseUtterance
from .aiUtil import AI_META_KEYS, LEGACY_AI_KEYS, as_dict, split_ai_fields
from .speaker import Speaker


class Utterance(BaseUtterance):
    """
    Represents a single utterance in the dataset.

    Takes the same arguments as convokit.Utterance, plus:

    :param ai_meta: dictionary of AI-specific attributes of the utterance (e.g. generation parameters)

    If `meta` contains an "ai_meta" entry (as in a dumped corpus) and `ai_meta` is not given,
    it is moved out of `meta` into the attribute.

    :ivar ai_meta: AI-specific attributes of the utterance. Assigning replaces the existing ai_meta.
    """

    def __init__(
        self,
        owner=None,
        id: Optional[str] = None,
        speaker: Optional[Speaker] = None,
        conversation_id: Optional[str] = None,
        reply_to: Optional[str] = None,
        timestamp: Optional[int] = None,
        text: str = "",
        meta: Optional[Dict] = None,
        ai_meta: Optional[Dict] = None,
    ):
        meta, ai_fields = split_ai_fields(meta, "utterance", strip_legacy=False)
        super().__init__(
            owner=owner,
            id=id,
            speaker=speaker,
            conversation_id=conversation_id,
            reply_to=reply_to,
            timestamp=timestamp,
            text=text,
            meta=meta,
        )
        self.ai_meta = ai_meta if ai_meta is not None else ai_fields.get("ai_meta")

    @classmethod
    def _from_base(cls, utt: BaseUtterance) -> "Utterance":
        """
        Convert a convokit.Utterance into a convokitai Utterance in place, moving AI fields out of its metadata.
        Metadata deletion must be unlocked by the caller if the utterance has an owner.
        """
        utt.__class__ = cls
        utt._ai_meta = as_dict(utt.meta.get("ai_meta"))
        for key in set(LEGACY_AI_KEYS) | set(AI_META_KEYS["utterance"]):
            if key in utt.meta:
                del utt.meta[key]
        return utt

    @property
    def ai_meta(self) -> Dict:
        return getattr(self, "_ai_meta", {})

    @ai_meta.setter
    def ai_meta(self, value):
        self._ai_meta = value or {}
