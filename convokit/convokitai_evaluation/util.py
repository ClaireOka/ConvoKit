"""
Helpers shared by the evaluation transformers. They read the ConvoKit AI format (see
``convokitai/FORMAT.md``): the mediator is the speaker whose ``ai_meta["role"]`` is
"public assistant", and ``Conversation.alias`` maps speaker ids to the names used in the chat.
"""

from typing import Callable, Dict, Optional

from convokit.model import Conversation, Speaker

MEDIATOR_ROLE = "public assistant"


def default_is_mediator(speaker: Speaker) -> bool:
    """Whether the speaker is the mediator: its ConvoKit AI role is "public assistant"."""
    return getattr(speaker, "ai_meta", {}).get("role") == MEDIATOR_ROLE


def default_speaker_name(conversation: Conversation, speaker: Speaker) -> str:
    """The name a speaker is addressed by in the conversation (``conversation.alias``), else its id."""
    return getattr(conversation, "alias", {}).get(speaker.id, speaker.id)


def default_topic(conversation: Conversation) -> str:
    """The topic line shown to the model: ``meta["topic"]``, plus the debate ``meta["statement"]``."""
    topic = conversation.meta.get("topic") or ""
    statement = conversation.meta.get("statement")
    return f"{topic} — debate statement: {statement}" if statement else topic


def pre_survey_text(conversation: Conversation, speaker: Speaker) -> Optional[str]:
    """
    The reason a participant gave for their stance before the conversation, from
    ``conversation.meta["pre_survey"]`` (a dict of speaker id -> text). None if there is none.
    """
    return (conversation.meta.get("pre_survey") or {}).get(speaker.id)


def display_names(
    conversation: Conversation,
    is_mediator: Callable[[Speaker], bool],
    name_func: Callable[[Conversation, Speaker], str],
) -> Dict[str, str]:
    """
    Map each non-mediator speaker id in the conversation to its name, in order of first utterance.
    """
    names = {}
    for utt in conversation.get_chronological_utterance_list():
        if utt.speaker.id not in names and not is_mediator(utt.speaker):
            names[utt.speaker.id] = name_func(conversation, utt.speaker)
    return names
