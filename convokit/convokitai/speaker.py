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

    def generate(
        self,
        conversation=None,
        reply_to: Optional[str] = None,
        append: bool = False,
        client=None,
        config_manager=None,
        id: Optional[str] = None,
        timestamp: Optional[int] = None,
    ):
        """
        Generate an Utterance for this speaker with convokit.genai, using the prompt in ai_meta["config"]
        (see convokitai.generation for the recognized config keys). Requires is_ai to be True.

        :param conversation: if given, the conversation transcript (up to `reply_to`) is included in the prompt
        :param reply_to: id of the utterance to reply to (defaults to the last utterance of `conversation`)
        :param append: whether to add the generated utterance to `conversation` (and its Corpus)
        :param client: a convokit.genai LLMClient to use instead of building one from the config
        :param config_manager: GenAIConfigManager used to build the client (defaults to ~/.convokit/config.yml)
        :param id: id of the generated utterance (random if not given)
        :param timestamp: timestamp of the generated utterance (defaults to after the conversation's last message)
        :return: the generated Utterance
        """
        from . import generation
        from .utterance import Utterance

        config = self.ai_meta.get("config") or {}
        if not self.is_ai:
            raise ValueError("Speaker {!r} is not an AI speaker (is_ai is False)".format(self.id))
        if not config.get("prompt"):
            raise ValueError("Speaker {!r} has no prompt in ai_meta['config']".format(self.id))
        if append and conversation is None:
            raise ValueError("A conversation is required to append the generated utterance")

        transcript = None
        name = self.id
        if conversation is not None:
            name = conversation.alias.get(self.id, self.id)
            if reply_to is None:
                reply_to = generation.last_utterance_id(conversation)
            if reply_to is not None:
                transcript = conversation.get_transcript(
                    until=conversation.get_utterance(reply_to)
                )
        prompt = generation.build_prompt(
            config["prompt"],
            transcript,
            "Write the next message in the conversation as {}. "
            "Respond with only the text of the message.".format(name),
        )
        text = generation.call_llm(generation.get_client(config, client, config_manager), config, prompt)

        utterance = Utterance(
            id=id or generation.new_id(),
            speaker=self,
            conversation_id=conversation.id if conversation is not None else None,
            reply_to=reply_to,
            timestamp=timestamp if timestamp is not None else generation.next_timestamp(conversation),
            text=text,
            ai_meta={"config": dict(config)},
        )
        if append:
            conversation.owner.add_utterances([utterance])
            utterance = conversation.owner.get_utterance(utterance.id)
        return utterance

    def __str__(self):
        return "Speaker(id: {}, is_ai: {}, vectors: {}, meta: {}, ai_meta: {})".format(
            repr(self.id), self.is_ai, self.vectors, self.meta, self.ai_meta
        )
