from typing import Dict, Iterable, List, Optional, Union


class Support:
    """
    Represents a single message from an Assistant to a Speaker, outside the main Conversation.

    :param id: the unique id of the support message
    :param text: text of the support message
    :param reply_to: text of the speaker message this support replies to, if any
    :param speaker_id: id of the Speaker being supported
    :param assistant_id: id of the Assistant that produced the message
    :param timestamp: timestamp of the support message
    :param conversation_id: id of the Conversation the support message belongs to
    :param meta: arbitrary dictionary of attributes associated with the support message
    """

    def __init__(
        self,
        id: str,
        text: str,
        reply_to: Optional[str] = None,
        speaker_id: Optional[str] = None,
        assistant_id: Optional[str] = None,
        timestamp: Union[str, int, None] = None,
        conversation_id: Optional[str] = None,
        meta: Optional[Dict] = None,
    ):
        self.id = id
        self.text = text
        self.reply_to = reply_to
        self.speaker_id = speaker_id
        self.assistant_id = assistant_id
        self.timestamp = timestamp
        self.conversation_id = conversation_id
        self.meta = dict(meta or {})

    def to_dict(self) -> Dict:
        payload = {
            "id": self.id,
            "text": self.text,
            "reply_to": self.reply_to,
            "speaker_id": self.speaker_id,
            "assistant_id": self.assistant_id,
            "timestamp": self.timestamp,
            "conversation_id": self.conversation_id,
        }
        if self.meta:
            payload["meta"] = dict(self.meta)
        return payload

    @classmethod
    def from_dict(cls, data: Dict) -> Optional["Support"]:
        if not isinstance(data, dict):
            return None
        return cls(
            id=data.get("id"),
            text=data.get("text", ""),
            reply_to=data.get("reply_to"),
            speaker_id=data.get("speaker_id"),
            assistant_id=data.get("assistant_id"),
            timestamp=data.get("timestamp"),
            conversation_id=data.get("conversation_id"),
            meta=data.get("meta") or {},
        )

    @staticmethod
    def normalize_list(value: Optional[Iterable]) -> List["Support"]:
        """
        Convert a list of Supports and/or dicts into a list of Supports. Unparseable entries are dropped.
        """
        normalized = []
        for item in value or []:
            if isinstance(item, Support):
                normalized.append(item)
            elif isinstance(item, dict):
                normalized.append(Support.from_dict(item))
        return normalized

    def __repr__(self):
        return "Support({})".format(self.to_dict())
