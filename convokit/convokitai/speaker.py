from typing import Dict, Optional

from convokit.model import Speaker as BaseSpeaker
from .aiUtil import AI_META_KEYS, LEGACY_AI_KEYS, as_dict, split_ai_fields


class Speaker(BaseSpeaker):
    """
    Represents a single speaker in a dataset, which may be a human or an AI.

    Takes the same arguments as convokit.Speaker, plus:

    :param is_ai: whether this speaker is known to be an LLM agent
    :param ai_meta: metadata for ConvoKitAI (empty if is_ai is False). Recognized keys:

        - "config": dict configuration that generates messages for this speaker
        - "role": the speaker's role in the conversation, e.g. "participant", "mediator"

    If `meta` contains "is_ai" / "ai_meta" entries (as in a dumped corpus) and the
    corresponding arguments are not given, they are moved out of `meta` into the attributes.
    A legacy "role" entry in `meta` is moved into ai_meta["role"].

    :ivar is_ai: whether this speaker is known to be an LLM agent
    :ivar ai_meta: metadata for ConvoKitAI. Assigning a dict merges it into the existing ai_meta.
    """

    def __init__(
        self,
        owner=None,
        id: str = None,
        utts=None,
        convos=None,
        meta: Optional[Dict] = None,
        is_ai: Optional[bool] = None,
        ai_meta: Optional[Dict] = None,
    ):
        legacy_role = (meta or {}).get("role")
        meta, ai_fields = split_ai_fields(meta, "speaker")
        super().__init__(owner=owner, id=id, utts=utts, convos=convos, meta=meta)
        self._ai_meta = {}
        self.is_ai = is_ai if is_ai is not None else ai_fields.get("is_ai", False)
        self.ai_meta = ai_meta if ai_meta is not None else ai_fields.get("ai_meta")
        self._migrate_legacy_role(legacy_role)

    @classmethod
    def _from_base(cls, speaker: BaseSpeaker) -> "Speaker":
        """
        Convert a convokit.Speaker into a convokitai Speaker in place, moving AI fields out of its metadata.
        Metadata deletion must be unlocked by the caller if the speaker has an owner.
        """
        speaker.__class__ = cls
        speaker._is_ai = bool(speaker.meta.get("is_ai", False))
        speaker._ai_meta = as_dict(speaker.meta.get("ai_meta"))
        speaker._migrate_legacy_role(speaker.meta.get("role"))
        for key in set(LEGACY_AI_KEYS) | set(AI_META_KEYS["speaker"]):
            if key in speaker.meta:
                del speaker.meta[key]
        return speaker

    def _migrate_legacy_role(self, role) -> None:
        if role is not None and "role" not in self.ai_meta:
            self.ai_meta = {"role": role}

    @property
    def is_ai(self) -> bool:
        return getattr(self, "_is_ai", False)

    @is_ai.setter
    def is_ai(self, value):
        self._is_ai = bool(value)

    @property
    def ai_meta(self) -> Dict:
        return as_dict(getattr(self, "_ai_meta", {}))

    @ai_meta.setter
    def ai_meta(self, value):
        if isinstance(value, dict):
            self._ai_meta = {**as_dict(getattr(self, "_ai_meta", {})), **value}
        else:
            self._ai_meta = {}

    def __str__(self):
        return "Speaker(id: {}, is_ai: {}, vectors: {}, meta: {}, ai_meta: {})".format(
            repr(self.id), self.is_ai, self.vectors, self.meta, self.ai_meta
        )
