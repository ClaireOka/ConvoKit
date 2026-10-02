"""
Mediator redirection: for each mediator turn, did it change what the participants said next?

The score comes from a language model's probabilities. Take a mediator turn T_k and the
participant reply that followed it. Ask the model how likely that reply was given T_k, and how
likely it was given a stand-in instead (the mediator's previous turn, or nothing). The
redirection score is the log-odds difference: positive means the reply was more expected after
T_k than after the stand-in, i.e. the turn steered the conversation; near zero means the reply
would have come anyway.

This is convokit.redirection (Nguyen et al. 2024) with context selectors for a mediator and
several participants instead of two alternating speakers:

    actual     [participant turn before T_k, T_k]      -> reply
    reference  [participant turn before T_k, T_(k-1)]  -> same reply   (counterfactual "prev")
               [participant turn before T_k]           -> same reply   (counterfactual "delete")

Each line is prefixed with the speaker's name ("Goose: ..., Mediator: ..."), the name the mediator uses.

Reference paper: Taking a turn for the better: Conversation redirection throughout the course of mental-health therapy
https://aclanthology.org/2024.findings-emnlp.555/
"""

from typing import Callable, Dict, List, Tuple

import pandas as pd

from convokit import Transformer
from convokit.model import Conversation, Corpus, Speaker
from convokit.redirection.redirection import Redirection
from .util import default_is_mediator, default_speaker_name

MEDIATOR_ROLE = (
    "public assistant"  # the role stored in utt.meta["role"] for the mediator
)
MEDIATOR_LABEL = "Mediator"  # how the mediator is named in the text the model reads
ROLE_ATTRIBUTE_NAME = "role"  # the key convokit.redirection.preprocessing reads
COUNTERFACTUALS = ("prev", "delete")


def mediator_speaker_prefixes(roles: List[str]) -> Dict[str, str]:
    """
    Speaker prefixes: the mediator's role becomes "Mediator: ", every other role is used as is
    ("Goose: "), so the prefix matches the name the mediator says.
    """
    return {
        role: (MEDIATOR_LABEL if role == MEDIATOR_ROLE else role) + ": "
        for role in roles
    }


def mediator_previous_context_selector(
    convo: Conversation,
    counterfactual: str = "prev",
    role_attribute_name: str = "role",
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Actual and reference contexts for every mediator turn that has a participant turn before it
    and a previous mediator turn. Mediator turns directly following another mediator turn are
    skipped.

    :param convo: Conversation whose utterances carry ``meta[role_attribute_name]``
    :param counterfactual: "prev" replaces the mediator turn with its previous turn in the
        reference context; "delete" removes it
    :param role_attribute_name: utterance metadata attribute holding the role
    :return: dicts of utterance id -> actual context and utterance id -> reference context
    """
    if counterfactual not in COUNTERFACTUALS:
        raise ValueError(
            f"counterfactual must be one of {COUNTERFACTUALS}, got {counterfactual!r}"
        )
    actual_contexts, reference_contexts = {}, {}
    utts = list(convo.iter_utterances())
    roles = sorted({utt.meta[role_attribute_name] for utt in utts})
    role_to_prefix = mediator_speaker_prefixes(roles)

    prev_participant = None
    prev_mediator = None
    for i, utt in enumerate(utts):
        cur_role = utt.meta[role_attribute_name]
        if cur_role != MEDIATOR_ROLE:
            prev_participant = utt
            continue

        adjacent = i > 0 and utts[i - 1].meta[role_attribute_name] == MEDIATOR_ROLE
        if prev_participant is not None and prev_mediator is not None and not adjacent:
            prev_data = role_to_prefix[prev_participant.meta[role_attribute_name]]
            prev_data += prev_participant.text
            cur_data = role_to_prefix[MEDIATOR_ROLE] + utt.text
            ref_data = role_to_prefix[MEDIATOR_ROLE] + prev_mediator.text

            actual_contexts[utt.id] = [prev_data, cur_data]
            if counterfactual == "prev":
                reference_contexts[utt.id] = [prev_data, ref_data]
            else:
                reference_contexts[utt.id] = [prev_data]

        prev_mediator = utt

    return actual_contexts, reference_contexts


def mediator_future_context_selector(
    convo: Conversation, role_attribute_name: str = "role"
) -> Dict[str, List[str]]:
    """
    Future context for every mediator turn: the participant turn right after it (none if the next
    turn is the mediator's again or empty).

    :param convo: Conversation whose utterances carry ``meta[role_attribute_name]``
    :param role_attribute_name: utterance metadata attribute holding the role
    :return: dict of utterance id -> future context
    """
    future_contexts = {}
    utts = list(convo.iter_utterances())
    roles = sorted({utt.meta[role_attribute_name] for utt in utts})
    role_to_prefix = mediator_speaker_prefixes(roles)

    for i, utt in enumerate(utts[:-1]):
        if utt.meta[role_attribute_name] != MEDIATOR_ROLE:
            continue
        nxt = utts[i + 1]
        if nxt.meta[role_attribute_name] == MEDIATOR_ROLE or not nxt.text.strip():
            continue
        future_contexts[utt.id] = [
            role_to_prefix[nxt.meta[role_attribute_name]] + nxt.text
        ]

    return future_contexts


class MediatorRedirection(Transformer):
    """
    Scores each mediator turn by how much it redirected the participant reply that followed it
    (see the module docstring) and stores the score in ``utt.meta["redirection"]``. Mediator
    turns with no participant reply, or no participant turn before them, get no score.

    Usage::

        likelihood_model = GemmaLikelihoodModel(...)          # from convokit.redirection
        redirection = MediatorRedirection(likelihood_model)
        redirection.transform(corpus)
        redirection.summarize(corpus)   # mean score per conversation

    It also writes ``utt.meta["role"]`` ("public assistant" for the mediator, else the
    participant's name), the attribute convokit.redirection reads; in the scored text the
    mediator's lines are prefixed "Mediator: ". ``fit`` fine-tunes the likelihood model (with
    convokit.redirection's own "Speaker A/B" formatting); scoring works without it.

    The corpus must say who the mediator is and what each speaker is called in the chat; by
    default these come from the ConvoKit AI fields (``speaker.ai_meta["role"]`` and
    ``conversation.alias``), and can be replaced with ``is_mediator`` / ``name_func``.

    :param likelihood_model: a convokit.redirection LikelihoodModel (e.g. GemmaLikelihoodModel);
        scores can be computed with the off-the-shelf model, ``fit`` fine-tunes it
    :param counterfactual: "prev" or "delete" (see mediator_previous_context_selector)
    :param is_mediator: function from Speaker to whether it is the mediator
    :param name_func: function from (Conversation, Speaker) to a participant's speaker prefix
    :param redirection_attribute_name: name of the utterance metadata attribute to store scores in
    """

    def __init__(
        self,
        likelihood_model,
        counterfactual: str = "prev",
        is_mediator: Callable[[Speaker], bool] = default_is_mediator,
        name_func: Callable[[Conversation, Speaker], str] = default_speaker_name,
        redirection_attribute_name: str = "redirection",
    ):
        self.is_mediator = is_mediator
        self.name_func = name_func
        self.redirection_attribute_name = redirection_attribute_name
        self.redirection = Redirection(
            likelihood_model=likelihood_model,
            previous_context_selector=lambda convo: mediator_previous_context_selector(
                convo,
                counterfactual=counterfactual,
            ),
            future_context_selector=mediator_future_context_selector,
            redirection_attribute_name=redirection_attribute_name,
        )

    def label_roles(self, corpus: Corpus, selector: Callable[[Conversation], bool]):
        """
        Write each utterance's role into ``meta["role"]`` in the selected conversations:
        "public assistant" for the mediator, the speaker's name for a participant.
        """
        for convo in corpus.iter_conversations():
            if not selector(convo):
                continue
            for utt in convo.iter_utterances():
                utt.meta[ROLE_ATTRIBUTE_NAME] = (
                    MEDIATOR_ROLE
                    if self.is_mediator(utt.speaker)
                    else self.name_func(convo, utt.speaker)
                )

    def fit(
        self,
        corpus: Corpus,
        y=None,
        train_selector: Callable[[Conversation], bool] = lambda convo: True,
        val_selector: Callable[[Conversation], bool] = lambda convo: True,
    ):
        """
        Fine-tune the likelihood model on the selected conversations (see Redirection.fit).
        """
        self.label_roles(corpus, train_selector)
        self.label_roles(corpus, val_selector)
        self.redirection.fit(
            corpus, train_selector=train_selector, val_selector=val_selector
        )
        return self

    def transform(
        self,
        corpus: Corpus,
        selector: Callable[[Conversation], bool] = lambda convo: True,
        verbosity: int = 5,
    ):
        """
        Label roles and store the redirection score of each scoreable mediator turn in
        ``utt.meta[redirection_attribute_name]``.

        :param corpus: the Corpus to transform
        :param selector: function from Conversation to whether it should be scored
        :param verbosity: print progress every ``verbosity`` conversations
        :return: the Corpus, with scores added
        """
        self.label_roles(corpus, selector)
        self.redirection.transform(corpus, selector=selector, verbosity=verbosity)
        return corpus

    def summarize(
        self,
        corpus: Corpus,
        selector: Callable[[Conversation], bool] = lambda convo: True,
    ) -> pd.DataFrame:
        """
        Mean redirection per conversation.

        :param corpus: a Corpus that has been transformed
        :param selector: function from Conversation to whether it should be included
        :return: DataFrame with one row per conversation: ``n_scored``, ``redirection_mean``
        """
        rows = []
        for convo in corpus.iter_conversations():
            if not selector(convo):
                continue
            scores = [
                utt.meta[self.redirection_attribute_name]
                for utt in convo.iter_utterances()
                if utt.meta.get(self.redirection_attribute_name) is not None
            ]
            if scores:
                rows.append(
                    {
                        "conversation_id": convo.id,
                        "n_scored": len(scores),
                        "redirection_mean": sum(scores) / len(scores),
                    }
                )
        return pd.DataFrame(rows)
