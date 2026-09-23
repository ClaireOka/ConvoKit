from typing import Callable, Dict, Generator, List, Optional

from pandas import DataFrame

from convokit.model import Corpus as BaseCorpus
from .aiUtil import as_dict, split_ai_fields
from .assistant import Assistant
from .conversation import Conversation
from .corpus_helpers import (
    extract_corpus_ai_fields,
    migrate_legacy_assistants_and_supports,
    stash_ai_fields_in_meta,
    unstash_ai_fields_from_meta,
    upgrade_components,
)
from .support import Support
from .utterance import Utterance


class Corpus(BaseCorpus):
    """
    Represents a dataset of conversations that may involve AI speakers and AI assistants.
    Can be loaded from a folder or constructed from a list of utterances, like convokit.Corpus,
    and all of its Speakers, Utterances, and Conversations are convokitai objects.

    Takes the same arguments as convokit.Corpus (passed through as keyword arguments), plus:

    :param has_ai: True if there are AI speakers in the corpus. If neither given nor stored in a loaded
        corpus, it is computed from whether any Speaker has is_ai set.
    :param meta: initial corpus-level metadata

    :ivar has_ai: True if there are AI speakers in the corpus
    :ivar ai_meta: metadata for ConvoKitAI. Assigning a dict merges it into the existing ai_meta.

    Assistants and Supports belong to Conversations (see Conversation.ai_meta); iter_assistants() and
    iter_supports() iterate over them across the whole corpus.
    """

    def __init__(
        self,
        filename: Optional[str] = None,
        utterances: Optional[List[Utterance]] = None,
        has_ai: Optional[bool] = None,
        meta: Optional[Dict] = None,
        **kwargs,
    ):
        super().__init__(filename=filename, utterances=utterances, **kwargs)
        self._init_ai_fields(filename)
        if meta:
            self.meta.update(split_ai_fields(meta, "corpus")[0])
        if has_ai is not None:
            self.has_ai = has_ai

    def _init_ai_fields(self, filename: Optional[str] = None) -> None:
        """
        Convert all components to convokitai objects and move the corpus-level AI fields out of
        corpus.meta into attributes. Corpus-level assistants / supports from earlier iterations of
        the format are moved onto their conversations.
        """
        self._has_ai = None
        self._ai_meta = {}
        upgrade_components(self)
        fields = extract_corpus_ai_fields(self, filename)
        self.ai_meta = fields["ai_meta"]
        if fields["has_ai"] is not None:
            self.has_ai = fields["has_ai"]
        migrate_legacy_assistants_and_supports(
            self, fields["legacy_assistants"], fields["legacy_supports"]
        )

    @classmethod
    def load(cls, filename: str, **kwargs) -> "Corpus":
        """
        Load a ConvoKit AI Corpus from a folder (or utterances.jsonl / utterances.json file).
        Equivalent to Corpus(filename=filename, **kwargs).

        :param filename: path to the corpus folder
        :param kwargs: any other arguments accepted by the Corpus constructor
        :return: the loaded Corpus
        """
        return cls(filename=filename, **kwargs)

    @classmethod
    def _from_base(cls, corpus: BaseCorpus) -> "Corpus":
        """
        Convert a convokit.Corpus into a convokitai Corpus in place.
        """
        corpus.__class__ = cls
        corpus._init_ai_fields()
        return corpus

    @classmethod
    def reconnect_to_db(cls, db_collection_prefix: str, db_host: Optional[str] = None):
        result = super().reconnect_to_db(db_collection_prefix, db_host=db_host)
        result._init_ai_fields()
        return result

    ############################################################################
    ## AI properties
    ############################################################################

    @property
    def has_ai(self) -> bool:
        has_ai = getattr(self, "_has_ai", None)
        if has_ai is None:
            return any(speaker.is_ai for speaker in getattr(self, "speakers", {}).values())
        return has_ai

    @has_ai.setter
    def has_ai(self, value):
        self._has_ai = bool(value)

    @property
    def ai_meta(self) -> Dict:
        return getattr(self, "_ai_meta", {})

    @ai_meta.setter
    def ai_meta(self, value):
        if isinstance(value, dict):
            self._ai_meta = {**as_dict(getattr(self, "_ai_meta", {})), **value}
        else:
            self._ai_meta = {}

    def iter_assistants(
        self, selector: Callable[[Assistant], bool] = lambda assistant: True
    ) -> Generator[Assistant, None, None]:
        """
        Get the Assistants of all Conversations in the Corpus, with an optional selector that filters
        for Assistants that should be included.
        """
        for convo in self.iter_conversations():
            for assistant in convo.assistants:
                if selector(assistant):
                    yield assistant

    def get_assistant(self, assistant_id: str) -> Assistant:
        """
        Get the Assistant with the specified id. Raises a KeyError if there is no such assistant.
        """
        for assistant in self.iter_assistants(lambda a: a.id == assistant_id):
            return assistant
        raise KeyError(assistant_id)

    def iter_supports(
        self, selector: Callable[[Support], bool] = lambda support: True
    ) -> Generator[Support, None, None]:
        """
        Get the Supports of all Conversations in the Corpus, with an optional selector that filters
        for Supports that should be included.
        """
        for convo in self.iter_conversations():
            for support in convo.supports:
                if selector(support):
                    yield support

    def _inherit_ai_fields(self, *sources: "Corpus", conversations: bool = True) -> None:
        """
        Merge the AI fields of the source corpora into this one (later sources take precedence).
        If `conversations` is True, also copy ai_meta of Conversations with matching ids.
        """
        for source in sources:
            if not isinstance(source, Corpus):
                continue
            self.ai_meta = source.ai_meta
            if getattr(source, "_has_ai", None) is not None:
                self.has_ai = self.has_ai or source.has_ai
            if conversations:
                for convo in self.iter_conversations():
                    if source.has_conversation(convo.id):
                        convo.ai_meta = dict(source.get_conversation(convo.id).ai_meta)

    ############################################################################
    ## overrides of convokit.Corpus methods that would otherwise produce
    ## plain convokit objects
    ############################################################################

    def dump(
        self,
        name: str,
        base_path: Optional[str] = None,
        exclude_vectors: List[str] = None,
        force_version: int = None,
        overwrite_existing_corpus: bool = False,
        fields_to_skip=None,
    ) -> None:
        """
        Dumps the corpus, its metadata, and its AI fields to disk. The AI fields are stored in the
        metadata, so the result is a valid convokit corpus folder. See convokit.Corpus.dump for parameters.
        """
        upgrade_components(self)
        stash_ai_fields_in_meta(self)
        try:
            super().dump(
                name,
                base_path=base_path,
                exclude_vectors=exclude_vectors,
                force_version=force_version,
                overwrite_existing_corpus=overwrite_existing_corpus,
                fields_to_skip=fields_to_skip,
            )
        finally:
            unstash_ai_fields_from_meta(self)

    def add_utterances(
        self, utterances: List[Utterance], warnings: bool = False, with_checks=True
    ) -> "Corpus":
        super().add_utterances(utterances, warnings=warnings, with_checks=with_checks)
        upgrade_components(self)
        return self

    @staticmethod
    def filter_utterances(source_corpus: "Corpus", selector: Callable[[Utterance], bool]):
        new_corpus = Corpus._from_base(BaseCorpus.filter_utterances(source_corpus, selector))
        new_corpus._inherit_ai_fields(source_corpus)
        return new_corpus

    @staticmethod
    def reindex_conversations(
        source_corpus: "Corpus",
        new_convo_roots: List[str],
        preserve_corpus_meta: bool = True,
        preserve_convo_meta: bool = True,
        verbose=True,
    ) -> "Corpus":
        new_corpus = Corpus._from_base(
            BaseCorpus.reindex_conversations(
                source_corpus,
                new_convo_roots,
                preserve_corpus_meta=preserve_corpus_meta,
                preserve_convo_meta=preserve_convo_meta,
                verbose=verbose,
            )
        )
        if preserve_corpus_meta:
            new_corpus._inherit_ai_fields(source_corpus, conversations=False)
        if preserve_convo_meta and isinstance(source_corpus, Corpus):
            for convo in new_corpus.iter_conversations():
                original_id = convo.meta.get("original_convo_id")
                if original_id is not None and source_corpus.has_conversation(original_id):
                    convo.ai_meta = dict(source_corpus.get_conversation(original_id).ai_meta)
        return new_corpus

    @staticmethod
    def merge(primary: "Corpus", secondary: "Corpus", warnings: bool = True):
        new_corpus = Corpus._from_base(BaseCorpus.merge(primary, secondary, warnings=warnings))
        new_corpus._inherit_ai_fields(primary, secondary)
        return new_corpus

    @staticmethod
    def from_pandas(
        utterances_df: DataFrame,
        speakers_df: Optional[DataFrame] = None,
        conversations_df: Optional[DataFrame] = None,
    ) -> "Corpus":
        return Corpus._from_base(
            BaseCorpus.from_pandas(utterances_df, speakers_df, conversations_df)
        )

    def print_summary_stats(self) -> None:
        """
        Helper function for printing the number of Speakers (and AI Speakers), Utterances, Conversations,
        Assistants, and Supports in this Corpus

        :return: None
        """
        super().print_summary_stats()
        print("Number of AI Speakers: {}".format(sum(s.is_ai for s in self.iter_speakers())))
        print("Number of Assistants: {}".format(sum(1 for _ in self.iter_assistants())))
        print("Number of Supports: {}".format(sum(1 for _ in self.iter_supports())))
