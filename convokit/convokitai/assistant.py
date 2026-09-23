from typing import Dict, Iterable, Optional


class Assistant:
    """
    Represents an AI assistant that supports one or more speakers in a Conversation.

    :param id: the unique id of the assistant
    :param config: arbitrary configuration of the assistant (e.g. model name, temperature)
    :param prompt: the prompt the assistant was run with
    :param conversation_id: id of the Conversation the assistant was attached to
    :param speakers: ids of the Speakers the assistant supports
    :param supports: Supports produced by this assistant (stored as dicts)
    """

    def __init__(
        self,
        id: str,
        config: Optional[Dict] = None,
        prompt: Optional[str] = None,
        conversation_id: Optional[str] = None,
        speakers: Optional[Iterable[str]] = None,
        supports: Optional[Iterable["Support"]] = None,
    ):
        self.id = id
        self.config = config or {}
        self.prompt = prompt or ""
        self.conversation_id = conversation_id
        self.speakers = list(speakers or [])
        self.supports = [s.to_dict() if hasattr(s, "to_dict") else s for s in (supports or [])]

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "config": self.config,
            "prompt": self.prompt,
            "conversation_id": self.conversation_id,
            "speakers": list(self.speakers),
            "supports": list(self.supports),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> Optional["Assistant"]:
        if not isinstance(data, dict):
            return None
        return cls(
            id=data.get("id"),
            config=data.get("config") or {},
            prompt=data.get("prompt"),
            conversation_id=data.get("conversation_id"),
            speakers=data.get("speakers") or [],
            supports=data.get("supports") or [],
        )

    def __repr__(self):
        return "Assistant({})".format(self.to_dict())
