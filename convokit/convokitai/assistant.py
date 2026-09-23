from typing import Dict, Iterable, List, Optional

from .support import Support


class Assistant:
    """
    Represents an AI assistant that sends Supports to one or more speakers in a Conversation.

    :param id: the unique id of the assistant
    :param config: configuration of the assistant, e.g. {"model": "gemini-2.5-flash", "temperature": 0.7}
    :param supports: the Supports (or their dict forms) produced by this assistant
    :param speakers: ids of the Speakers that can see this assistant's Supports
    :param prompt: the prompt that generates a support message from this assistant
    :param conversation_id: id of the Conversation the assistant belongs to
    """

    def __init__(
        self,
        id: str,
        config: Optional[Dict] = None,
        supports: Optional[Iterable] = None,
        speakers: Optional[Iterable[str]] = None,
        prompt: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ):
        self.id = id
        self.config = config or {}
        self.supports: List[Support] = Support.normalize_list(supports)
        self.speakers = list(speakers or [])
        self.prompt = prompt or ""
        self.conversation_id = conversation_id

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "config": self.config,
            "supports": [support.to_dict() for support in self.supports],
            "speakers": list(self.speakers),
            "prompt": self.prompt,
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
            prompt=data.get("prompt"),
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

    def __repr__(self):
        return "Assistant({})".format(self.to_dict())
