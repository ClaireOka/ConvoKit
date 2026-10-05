ConvoKit AI
===========

ConvoKit AI (the ``convokitai`` package) extends the ConvoKit data model to represent
conversations that involve AI agents. On top of the regular :doc:`Corpus model <model>`, it adds:

* **AI speakers**: speakers marked with ``is_ai``, together with the configuration (prompt and
  model) that generates their messages.
* **Private assistants** (:class:`~convokitai.private_assistant.PrivateAssistant`): AI assistants that
  privately help one or more speakers, outside the conversation itself.
* **Supports** (:class:`~convokitai.support.Support`): the messages a private assistant sends to the
  speaker(s) it helps.

``convokitai`` mirrors the ``convokit.model`` API, so ``convokitai.Corpus`` can be used anywhere a
``convokit.Corpus`` is expected, including with all ConvoKit transformers. Every ConvoKit AI corpus is
also a valid ConvoKit corpus: the AI fields are stored inside the regular metadata on disk, so
``convokit.Corpus`` can load it too.

Two companion packages build on this format:

* :doc:`ConvoKit AI Simulation <convokitAISimulation>` runs AI agent conversations on
  `Deliberate Lab <https://github.com/PAIR-code/deliberate-lab>`_ and returns them as a ConvoKit AI corpus.
* :doc:`ConvoKit AI Evaluation <convokitAIEvaluation>` provides transformers that evaluate
  AI-mediated conversations.

Installation
------------

``convokitai`` is installed together with ConvoKit and imported as a separate top-level package::

    import convokitai

To install it on its own, without the rest of ConvoKit's source::

    pip install "git+https://github.com/CornellNLP/ConvoKit.git@master#subdirectory=convokit/convokitai"

Quick start
-----------

.. code-block:: python

    from convokitai import Corpus, download

    corpus = Corpus(filename=download("llm-facilitation-corpus"))
    corpus.print_summary_stats()  # also counts AI speakers, private assistants and supports

    convo = next(corpus.iter_conversations())

    # AI speakers, and the names speakers go by in the conversation
    for speaker in convo.iter_speakers():
        print(convo.alias.get(speaker.id, speaker.id), speaker.is_ai, speaker.ai_meta.get("role"))

    # the conversation, with each private assistant's supports shown under the message they reply to
    print(convo.get_transcript(supports=True))

    # all supports across the corpus
    for support in corpus.iter_supports():
        print(support.private_assistant_id, support.text)

AI speakers and private assistants with a `generation config`_ can also produce new messages with
:doc:`convokit.genai <genai>` (the API key for the model's provider must be configured; see
:doc:`GenAI <genai>`):

.. code-block:: python

    # the next message from an AI speaker, given the conversation so far
    utterance = speaker.generate(conversation=convo, append=True)

    # a private assistant's support for the speaker it helps, given what they have drafted
    support = assistant.generate(conversation=convo, draft="I think we should")

Data format
-----------

The fields below are added to the standard :doc:`ConvoKit data format <data_format>`. In memory they
are attributes of each object; on disk they are stored in its ``meta`` (``meta["is_ai"]``,
``meta["ai_meta"]``, ...).

Speaker
^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Field
     - Type
     - Description
   * - ``is_ai``
     - ``bool``
     - Whether the speaker is known to be an AI agent.
   * - ``ai_meta["config"]``
     - ``dict``
     - The `generation config`_ that produces this speaker's messages (used by ``Speaker.generate``).
   * - ``ai_meta["role"]``
     - ``str``
     - The speaker's role in the conversation, e.g. ``"participant"`` or ``"public assistant"``
       (a mediator who speaks in the conversation).

Utterance
^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Field
     - Type
     - Description
   * - ``ai_meta["config"]``
     - ``dict``
     - The generation config used to produce this message. Empty for human utterances.
   * - ``ai_meta["supports"]``
     - ``list[Support]``
     - The supports sent in reply to this utterance (also available as ``Utterance.supports``).

Other outputs of the model call can be stored in ``ai_meta`` as well, for example a mediator's
stated reason for its message under ``"explanation"``.

Conversation
^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Field
     - Type
     - Description
   * - ``ai_meta["alias"]``
     - ``dict[str, str]``
     - The name each speaker goes by in the conversation (e.g. ``"Bear"``), keyed by speaker ID.
       Also available as ``Conversation.alias``.
   * - ``ai_meta["private_assistants"]``
     - ``list[PrivateAssistant]``
     - The private assistants in this conversation (``Conversation.private_assistants``).
   * - ``ai_meta["supports"]``
     - ``list[Support]``
     - Optional. Supports stored at the conversation level. ``Conversation.supports`` returns these
       together with the supports of the private assistants and of the utterances.

Corpus
^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Field
     - Type
     - Description
   * - ``has_ai``
     - ``bool``
     - Whether any speaker in the corpus is AI. Computed from the speakers if not given.
   * - ``ai_meta``
     - ``dict``
     - Free-form corpus-level metadata. For example, ConvoKit AI Simulation stores the simulation
       configuration under ``"simulation_config"``.

PrivateAssistant
^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Field
     - Type
     - Description
   * - ``id``
     - ``str``
     - Unique ID of the private assistant.
   * - ``config``
     - ``dict``
     - The `generation config`_ that produces its supports (used by ``PrivateAssistant.generate``).
   * - ``supports``
     - ``list[Support]``
     - The supports this assistant sent.
   * - ``speakers``
     - ``list[str]``
     - IDs of the speakers who can see this assistant's supports.
   * - ``conversation_id``
     - ``str``
     - ID of the conversation the assistant belongs to.

Support
^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Field
     - Type
     - Description
   * - ``id``
     - ``str``
     - Unique ID of the support.
   * - ``text``
     - ``str``
     - The message the assistant sent.
   * - ``reply_to``
     - ``str``
     - ID of the utterance the assisted speaker was replying to.
   * - ``draft``
     - ``str``
     - What the assisted speaker had drafted so far (``""`` if nothing).
   * - ``private_assistant_id``
     - ``str``
     - ID of the private assistant that sent the support.
   * - ``timestamp``
     - ``int`` or ``str``
     - When the support was sent.

Generation config
^^^^^^^^^^^^^^^^^

AI speakers and private assistants that can generate new messages carry a generation config:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Key
     - Type
     - Description
   * - ``prompt``
     - ``str``
     - The prompt that produces a message (required).
   * - ``model``
     - ``str``
     - The model name, e.g. ``"gemini-2.5-flash"`` or ``"gpt-4o-mini"``.
   * - ``provider``
     - ``str``
     - Optional. The ``convokit.genai`` provider: ``"gemini"``, ``"gpt"`` or ``"local"``. Inferred
       from ``model`` if missing.
   * - ``temperature``
     - ``float``
     - Optional. Sampling temperature.

For example:

.. code-block:: python

    {
        "prompt": "You help Goose phrase their replies politely.",
        "model": "gemini-2.5-flash",
        "temperature": 0.7,
    }

API reference
-------------

Corpus
^^^^^^

.. automodule:: convokitai.corpus
    :members:

Conversation
^^^^^^^^^^^^

.. automodule:: convokitai.conversation
    :members:

Speaker
^^^^^^^

.. automodule:: convokitai.speaker
    :members:

Utterance
^^^^^^^^^

.. automodule:: convokitai.utterance
    :members:

PrivateAssistant
^^^^^^^^^^^^^^^^

.. automodule:: convokitai.private_assistant
    :members:

Support
^^^^^^^

.. automodule:: convokitai.support
    :members:
