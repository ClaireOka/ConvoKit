"""
ConvoKit AI: ConvoKit's data model extended with AI speakers, assistants, and support messages.

Mirrors the convokit.model API, so it can be used as a drop-in replacement::

    import convokitai
    corpus = convokitai.Corpus.load("path/to/corpus")   # or convokitai.Corpus(filename=...)
"""

from convokit.model import ConvoKitIndex, ConvoKitMatrix, ConvoKitMeta, CorpusComponent, UtteranceNode
from convokit.util import download
from .assistant import Assistant
from .conversation import Conversation
from .corpus import Corpus
from .speaker import Speaker
from .support import Support
from .utterance import Utterance

__all__ = [
    "Assistant",
    "Conversation",
    "ConvoKitIndex",
    "ConvoKitMatrix",
    "ConvoKitMeta",
    "Corpus",
    "CorpusComponent",
    "Speaker",
    "Support",
    "Utterance",
    "UtteranceNode",
    "download",
]
