from typing import Dict, Iterable, List, Optional

from .support import Support


class Assistant:
    """
    Represents an AI assistant that sends Supports to one or more speakers in a Conversation.

    :param id: the unique id of the assistant
    :param config: configuration of the assistant, e.g.
        {"prompt": "...", "model": "gemini-2.5-flash", "temperature": 0.7}. config["prompt"] is the
        prompt that generates a support message from this assistant (required by generate()).
    :param supports: the Supports (or their dict forms) produced by this assistant
    :param speakers: ids of the Speakers that can see this assistant's Supports
    :param conversation_id: id of the Conversation the assistant belongs to
    """

    def __init__(
        self,
        id: str,
        config: Optional[Dict] = None,
        supports: Optional[Iterable] = None,
        speakers: Optional[Iterable[str]] = None,
        conversation_id: Optional[str] = None,
    ):
        self.id = id
        self.config = config or {}
        self.supports: List[Support] = Support.normalize_list(supports)
        self.speakers = list(speakers or [])
        self.conversation_id = conversation_id

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "config": self.config,
            "supports": [support.to_dict() for support in self.supports],
            "speakers": list(self.speakers),
            "conversation_id": self.conversation_id,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> Optional["Assistant"]:
        if not isinstance(data, dict):
            return None
        return cls(
            id=data.get("id"),
            config=data.get("config") or {},
            supports=data.get("supports") or [],
            speakers=data.get("speakers") or [],
            conversation_id=data.get("conversation_id"),
        )

    @staticmethod
    def normalize_list(value) -> List["Assistant"]:
        """
        Convert a list (or dict of id -> value) of Assistants and/or dicts into a list of Assistants.
        Unparseable entries are dropped.
        """
        if isinstance(value, dict):
            value = value.values()
        normalized = []
        for item in value or []:
            if isinstance(item, Assistant):
                normalized.append(item)
            elif isinstance(item, dict):
                normalized.append(Assistant.from_dict(item))
        return normalized

    def generate(
        self,
        conversation=None,
        reply_to: Optional[str] = None,
        draft: str = "",
        append: bool = False,
        client=None,
        config_manager=None,
        id: Optional[str] = None,
        timestamp: Optional[int] = None,
    ) -> Support:
        """
        Generate a Support from this assistant with convokit.genai, using the prompt in config["prompt"]
        (see convokitai.generation for the recognized config keys).

        :param conversation: if given, the conversation transcript (with Supports) up to `reply_to` is
            included in the prompt, along with earlier Supports replying to `reply_to`
        :param reply_to: id of the utterance the assisted speaker is replying to (defaults to the last
            utterance of `conversation`)
        :param draft: the draft the assisted speaker has written so far
        :param append: whether to add the generated support to this assistant and `conversation`
        :param client: a convokit.genai LLMClient to use instead of building one from the config
        :param config_manager: GenAIConfigManager used to build the client (defaults to ~/.convokit/config.yml)
        :param id: id of the generated support (random if not given)
        :param timestamp: timestamp of the generated support (defaults to after the conversation's last message)
        :return: the generated Support
        """
        from . import generation

        config = self.config
        if not config.get("prompt"):
            raise ValueError("Assistant {!r} has no prompt in config".format(self.id))
        if append and conversation is None:
            raise ValueError("A conversation is required to append the generated support")

        transcript = None
        viewers = list(self.speakers)
        if conversation is not None:
            viewers = [conversation.alias.get(s, s) for s in self.speakers]
            if reply_to is None:
                reply_to = generation.last_utterance_id(conversation)
            if reply_to is not None:
                # end with the last support already given for this reply, if any
                until = generation.last_support(conversation, reply_to) or conversation.get_utterance(
                    reply_to
                )
                transcript = conversation.get_transcript(supports=True, until=until)

        instruction = ""
        if draft:
            instruction += "The draft of the reply written so far:\n{}\n\n".format(draft)
        instruction += "Write your support message{}. Respond with only the text of the message.".format(
            " to " + ", ".join(viewers) if viewers else ""
        )
        text = generation.call_llm(
            generation.get_client(config, client, config_manager),
            config,
            generation.build_prompt(config["prompt"], transcript, instruction),
        )

        support = Support(
            id=id or generation.new_id(),
            text=text,
            reply_to=reply_to,
            draft=draft,
            assistant_id=self.id,
            timestamp=timestamp if timestamp is not None else generation.next_timestamp(conversation),
        )
        if append:
            self.supports.append(support)
            if self.conversation_id is None:
                self.conversation_id = conversation.id
            stored = next((a for a in conversation.assistants if a.id == self.id), None)
            if stored is None:
                conversation.assistants = conversation.assistants + [self]
            elif stored is not self:
                stored.supports.append(support)
            if "supports" in conversation.ai_meta:
                conversation.supports = conversation.supports + [support]
        return support

    def __repr__(self):
        return "Assistant({})".format(self.to_dict())
