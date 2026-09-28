from typing import Optional, Union, Callable, Dict, Any, List, Sequence
from convokit import Transformer, Corpus, Conversation, Speaker, Utterance
from .factory import get_llm_client
from .genai_config import GenAIConfigManager

try:
    from convokitai import Assistant, Support
except ImportError:
    Assistant = Support = None

OBJECT_LEVELS = ["conversation", "speaker", "utterance", "corpus", "assistant", "support"]

# object levels that only exist in a convokitai Corpus
AI_OBJECT_LEVELS = ["assistant", "support"]


class LLMPromptTransformer(Transformer):
    """
    A ConvoKit Transformer that uses GenAI clients to process objects and store outputs as metadata.

    This transformer applies LLM prompts to different levels of the corpus (conversation, speaker, utterance, corpus)
    using a formatter function to prepare the object data for the prompt, and stores the LLM responses as metadata.
    On a convokitai Corpus, it can also be applied to Assistants and Supports (their outputs are stored in
    assistant.meta / support.meta), and to several levels at once, e.g. ["speaker", "assistant"] or
    ["utterance", "support"].

    :param provider: LLM provider name ("gpt", "gemini", "local", etc.)
    :param model: LLM model name
    :param object_level: Object level at which to apply the transformer ("conversation", "speaker", "utterance",
        "corpus", "assistant", "support"), or a list of object levels
    :param prompt: Template string for the prompt. Must contain '{formatted_object}' as a placeholder where the formatted object data will be inserted
    :param formatter: Function that takes an object and returns a string representation that will replace the '{formatted_object}' placeholder in the prompt.
        Can also be a dict mapping each object level to its own formatter function
    :param metadata_name: Name of the metadata field to store the LLM response
    :param selector: Optional function to filter which objects to process. Defaults to processing all objects
    :param config_manager: GenAIConfigManager instance for LLM API key management
    :param llm_kwargs: Additional keyword arguments to pass to the LLM client
    """

    def __init__(
        self,
        provider: str,
        model: str,
        object_level: Union[str, Sequence[str]],
        prompt: str,
        formatter: Union[Callable[[Any], str], Dict[str, Callable[[Any], str]]],
        metadata_name: str,
        selector: Optional[Callable[[Any], bool]] = None,
        config_manager: Optional[GenAIConfigManager] = None,
        llm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.provider = provider
        self.model = model
        self.object_level = object_level
        self.object_levels: List[str] = (
            [object_level] if isinstance(object_level, str) else list(object_level)
        )
        self.prompt = prompt
        self.formatter = formatter
        self.metadata_name = metadata_name
        self.selector = selector or (lambda obj: True)
        self.config_manager = config_manager or GenAIConfigManager()
        self.llm_kwargs = llm_kwargs or {}

        if model is not None:
            self.llm_kwargs["model"] = model

        invalid_levels = [level for level in self.object_levels if level not in OBJECT_LEVELS]
        if invalid_levels or not self.object_levels:
            raise ValueError(
                f"Invalid object_level: {object_level}. Must be one of (or a list of): {', '.join(OBJECT_LEVELS)}"
            )

        if any(level in AI_OBJECT_LEVELS for level in self.object_levels) and Assistant is None:
            raise ImportError(
                "The assistant and support object levels require convokitai. Please install convokitai."
            )

        if isinstance(formatter, dict):
            missing_levels = [level for level in self.object_levels if level not in formatter]
            if missing_levels:
                raise ValueError(f"formatter has no entry for object level(s): {missing_levels}")

        if "{formatted_object}" not in prompt:
            raise ValueError(
                "Prompt must contain '{formatted_object}' placeholder for the formatted object data"
            )

        self.llm_client = get_llm_client(provider, self.config_manager, **self.llm_kwargs)

    def _object_level_of(self, obj) -> Optional[str]:
        """
        Get the object level of an object, or None if it is not a valid object.

        :param obj: Object to get the level of
        :return: Object level name
        """
        # Support / Assistant come first since they are not ConvoKit objects
        if Support is not None and isinstance(obj, Support):
            return "support"
        if Assistant is not None and isinstance(obj, Assistant):
            return "assistant"
        for level, cls in [
            ("utterance", Utterance),
            ("conversation", Conversation),
            ("speaker", Speaker),
            ("corpus", Corpus),
        ]:
            if isinstance(obj, cls):
                return level
        return None

    def _format_prompt(self, obj: Any) -> str:
        """
        Format the prompt with the object data using the formatter function.

        :param obj: Object to format
        :return: Formatted prompt string
        """
        try:
            formatter = self.formatter
            if isinstance(formatter, dict):
                formatter = formatter[self._object_level_of(obj)]
            formatted_object = formatter(obj)
            return self.prompt.format(formatted_object=formatted_object)
        except Exception as e:
            raise ValueError(f"Error formatting object for prompt: {e}")

    def _process_object(self, obj: Any) -> None:
        """
        Process a single object with the LLM and store the result in metadata.

        :param obj: Object to process
        """
        try:
            formatted_prompt = self._format_prompt(obj)
            response = self.llm_client.generate(formatted_prompt)
            obj.add_meta(self.metadata_name, response.text)
        except Exception as e:
            print(f"Error processing {self._object_level_of(obj)} {obj.id}: {e}")
            obj.add_meta(self.metadata_name, None)

    def _iter_objects(self, corpus: Corpus, object_level: str):
        """
        Iterate over the objects of the corpus at the given object level.

        :param corpus: The corpus to iterate over
        :param object_level: Object level to iterate over
        """
        if object_level == "utterance":
            yield from corpus.iter_utterances()
        elif object_level == "conversation":
            yield from corpus.iter_conversations()
        elif object_level == "speaker":
            yield from corpus.iter_speakers()
        elif object_level == "corpus":
            yield corpus
        else:
            if not hasattr(corpus, "iter_assistants"):
                raise ValueError(
                    f"The {object_level} object level requires a convokitai Corpus, got {type(corpus).__name__}"
                )
            if object_level == "assistant":
                yield from corpus.iter_assistants()
            else:
                # a Support can be stored in several places of a conversation, so skip repeated ids
                seen = set()
                for support in corpus.iter_supports():
                    if support.id is None or support.id not in seen:
                        seen.add(support.id)
                        yield support

    def transform(self, corpus: Corpus) -> Corpus:
        """
        Apply the GenAI transformer to the corpus.

        :param corpus: The corpus to transform
        :return: The transformed corpus with LLM responses added as metadata
        """
        for object_level in self.object_levels:
            for obj in self._iter_objects(corpus, object_level):
                if self.selector(obj):
                    self._process_object(obj)
                else:
                    obj.add_meta(self.metadata_name, None)

        return corpus

    def transform_single(self, obj: Any) -> Any:
        """
        Transform a single object (utterance, conversation, speaker, corpus, assistant, or support) with the LLM prompt.
        This method allows users to easily test their prompt on a single unit without processing an entire corpus.

        :param obj: The object to transform. Can be:
            - A string (will be converted to an Utterance with a default speaker)
            - An Utterance, Conversation, Speaker, Corpus, Assistant, or Support object
        :return: The transformed object with LLM response stored in metadata
        """
        # Handle string input by converting to Utterance
        if isinstance(obj, str):
            if "utterance" not in self.object_levels:
                raise ValueError(
                    f"Cannot convert string to {self.object_level}. String input is only supported for utterance-level transformation."
                )
            obj = Utterance(text=obj, speaker=Speaker(id="speaker"))

        # Validate object type matches the transformer's object_level
        if self._object_level_of(obj) not in self.object_levels:
            raise ValueError(
                f"Expected object of level {self.object_level} for transformation, got {type(obj).__name__}"
            )

        # Check if object passes the selector
        if not self.selector(obj):
            obj.add_meta(self.metadata_name, None)
            return obj

        # Process the object
        self._process_object(obj)
        return obj
