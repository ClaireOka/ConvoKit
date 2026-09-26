from typing import Dict, List, Optional

from convokit.model import Utterance as BaseUtterance
from .aiUtil import AI_META_KEYS, LEGACY_AI_KEYS, as_dict, split_ai_fields
from .speaker import Speaker
from .support import Support


class Utterance(BaseUtterance):
    """
    Represents a single utterance in the dataset.

    Takes the same arguments as convokit.Utterance, plus:

    :param ai_meta: additional outputs from the speaker LLM call for this message (empty if there are none
        or the speaker is not an agent). Recognized keys:

        - "config": dict configuration used to generate the message
        - "supports": list of Supports attached to this utterance

    If `meta` contains an "ai_meta" entry (as in a dumped corpus) and `ai_meta` is not given,
    it is moved out of `meta` into the attribute.

    :ivar ai_meta: AI metadata of the utterance. Assigning replaces the existing ai_meta.
    :ivar supports: the Supports attached to this utterance
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
        utt.ai_meta = as_dict(utt.meta.get("ai_meta"))
        for key in set(LEGACY_AI_KEYS) | set(AI_META_KEYS["utterance"]):
            if key in utt.meta:
                del utt.meta[key]
        return utt

    @property
    def ai_meta(self) -> Dict:
        return getattr(self, "_ai_meta", {})

    @ai_meta.setter
    def ai_meta(self, value):
        value = dict(value or {})
        if "supports" in value:
            value["supports"] = Support.normalize_list(value["supports"])
        self._ai_meta = value

    @property
    def supports(self) -> List[Support]:
        return Support.normalize_list(self.ai_meta.get("supports", []))

    @supports.setter
    def supports(self, value):
        self.ai_meta = {**self.ai_meta, "supports": value}

    def get_transcript(self, supports: bool = False) -> str:
        """
        Get a plain-text transcript of this utterance's Conversation, from its beginning up to and
        including this utterance. See Conversation.get_transcript.

        :param supports: whether to include Supports (Supports replying to this utterance come after it,
            so they are not included)
        :return: the transcript as a single string
        """
        if self.owner is None:
            raise ValueError("Utterance {!r} is not part of a Corpus".format(self.id))
        return self.get_conversation().get_transcript(supports=supports, until=self)
