"""
ConvoKit AI evaluation: metrics for AI-mediated conversations.

- CIG: per-utterance conversational information gain, judged by an LLM
- ClaimTracker: claim-level consensus between participants, and where their claims came from
- MediatorRedirection: redirection of each mediator turn, via convokit.redirection
"""

from .cig import CIG
from .claims import ClaimTracker
from .llm import GPT5Client
from .util import (
    default_is_mediator,
    default_speaker_name,
    default_topic,
    pre_survey_text,
)

MediatorRedirection = None
mediator_future_context_selector = mediator_previous_context_selector = None
try:
    from .redirection import (
        MediatorRedirection,
        mediator_future_context_selector,
        mediator_previous_context_selector,
    )
except ImportError:
    pass

__all__ = [
    "CIG",
    "ClaimTracker",
    "MediatorRedirection",
    "GPT5Client",
    "default_is_mediator",
    "default_speaker_name",
    "default_topic",
    "pre_survey_text",
    "mediator_future_context_selector",
    "mediator_previous_context_selector",
]
