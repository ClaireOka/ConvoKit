from typing import Callable, Dict, List, Optional

from pandas import DataFrame

from convokit.model import Corpus as BaseCorpus
from .aiUtil import as_dict, split_ai_fields
from .assistant import Assistant
from .conversation import Conversation
from .corpus_helpers import (
    distribute_supports_to_conversations,
    dump_supports,
    extract_corpus_ai_fields,
    get_dump_dirpath,
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

    :param has_ai: whether the corpus involves AI. If loading from a folder and not given, the stored value is used.
    :param assistants: dictionary of assistant id -> Assistant (or its dict form)
    :param supports: list of Supports (or their dict forms)
    :param meta: initial corpus-level metadata

    :ivar has_ai: whether the corpus involves AI
    :ivar ai_meta: AI-specific attributes of the corpus. Assigning a dict merges it into the existing ai_meta.
    :ivar assistants: dictionary of assistant id -> Assistant
    :ivar supports: list of all Supports in the corpus
    """

    def __init__(
        self,
        filename: Optional[str] = None,
        utterances: Optional[List[Utterance]] = None,
        has_ai: Optional[bool] = None,
        assistants: Optional[Dict] = None,
        supports: Optional[List] = None,
        meta: Optional[Dict] = None,
        **kwargs,
    ):
        super().__init__(filename=filename, utterances=utterances, **kwargs)
        self._init_ai_fields(filename)
        if meta:
            self.meta.update(split_ai_fields(meta, "corpus")[0])
        if has_ai is not None:
            self.has_ai = has_ai
        if assistants is not None:
            self.assistants = assistants
        if supports is not None:
            self.supports = supports
            distribute_supports_to_conversations(self, self.supports)

    def _init_ai_fields(self, filename: Optional[str] = None) -> None:
        """
        Convert all components to convokitai objects and move the corpus-level AI fields out of
        corpus.meta (and supports.json, if loading from a folder) into attributes.
        """
        self._has_ai = False
        self._ai_meta = {}
        self._supports = []
        upgrade_components(self)
        fields = extract_corpus_ai_fields(self, filename)
        self.ai_meta = fields["ai_meta"]
        self.has_ai = fields["has_ai"]
        self.assistants = fields["assistants"]
        self.supports = fields["supports"]
        distribute_supports_to_conversations(self, self.supports)

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
        return getattr(self, "_has_ai", False)

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

    @property
    def supports(self) -> List[Support]:
        return list(getattr(self, "_supports", []))

    @supports.setter
    def supports(self, value):
        self._supports = Support.normalize_list(value)

    @property
    def assistants(self) -> Dict[str, Assistant]:
        raw_assistants = as_dict(self.ai_meta.get("assistants"))
        return {
            k: Assistant.from_dict(v) if isinstance(v, dict) else v
            for k, v in raw_assistants.items()
        }

    @assistants.setter
    def assistants(self, value):
        payload = {k: v.to_dict() if isinstance(v, Assistant) else v for k, v in (value or {}).items()}
        self.ai_meta = {"assistants": payload}

    def _inherit_ai_fields(self, *sources: "Corpus", conversations: bool = True) -> None:
        """
        Merge the AI fields of the source corpora into this one (later sources take precedence).
        If `conversations` is True, also copy ai_meta of Conversations with matching ids.
        """
        for source in sources:
            if not isinstance(source, Corpus):
                continue
            assistants = {**self.assistants, **source.assistants}
            self.ai_meta = source.ai_meta
            self.assistants = assistants
            self.has_ai = self.has_ai or source.has_ai
            seen = {support.id for support in self.supports}
            self.supports = self.supports + [s for s in source.supports if s.id not in seen]
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
        Dumps the corpus, its metadata, and its AI fields to disk. The result is a valid convokit
        corpus folder with an additional supports.json file. See convokit.Corpus.dump for parameters.
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
            dump_supports(
                self.supports,
                get_dump_dirpath(self, name, base_path, overwrite_existing_corpus),
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
        print("Number of Assistants: {}".format(len(self.assistants)))
        print("Number of Supports: {}".format(len(self.supports)))
