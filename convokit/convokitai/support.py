from typing import Dict, Iterable, List, Optional, Union


class Support:
    """
    Represents a single message from an Assistant to the speaker(s) it assists, outside the main Conversation.
    The speakers that can see a Support are given by its Assistant's `speakers`.

    :param id: the unique id of the support
    :param text: output of the support
    :param reply_to: id of the utterance the assisted speaker is replying to
    :param draft: the draft of the post the assisted speaker has written so far ("" means the draft is empty)
    :param assistant_id: id of the Assistant that produced the support
    :param timestamp: the timestamp the support was sent
    """

    def __init__(
        self,
        id: str,
        text: str,
        reply_to: Optional[str] = None,
        draft: str = "",
        assistant_id: Optional[str] = None,
        timestamp: Union[str, int, None] = None,
    ):
        self.id = id
        self.text = text
        self.reply_to = reply_to
        self.draft = draft if draft is not None else ""
        self.assistant_id = assistant_id
        self.timestamp = timestamp

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "text": self.text,
            "reply_to": self.reply_to,
            "draft": self.draft,
            "assistant_id": self.assistant_id,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> Optional["Support"]:
        """
        Build a Support from its dict form. Keys outside the Support fields are ignored.
        """
        if not isinstance(data, dict):
            return None
        return cls(
            id=data.get("id"),
            text=data.get("text", ""),
            reply_to=data.get("reply_to"),
            draft=data.get("draft", ""),
            assistant_id=data.get("assistant_id"),
            timestamp=data.get("timestamp"),
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
