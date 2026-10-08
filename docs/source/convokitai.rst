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
  ConvoArena (built on `Deliberate Lab <https://github.com/PAIR-code/deliberate-lab>`_) and returns
  them as a ConvoKit AI corpus.
* :doc:`ConvoKit AI Evaluation <convokitAIEvaluation>` provides transformers that evaluate
  AI-mediated conversations.

Installation
------------

``convokitai`` is part of ConvoKit (version 5.0.0 and later), so installing ConvoKit also installs
it::

    pip install convokit

It is imported as a separate top-level package::

    import convokitai

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

AI speakers and private assistants with a :ref:`generation config <convokitai-generation-config>`
can also produce new messages with :doc:`convokit.genai <genai>` (the API key for the model's
provider must be configured; see :doc:`GenAI <genai>`):

.. code-block:: python

    # the next message from an AI speaker, given the conversation so far
    utterance = speaker.generate(conversation=convo, append=True)

    # a private assistant's support for the speaker it helps, given what they have drafted
    support = assistant.generate(conversation=convo, draft="I think we should")

Data format
-----------

ConvoKit AI adds fields for AI speakers, private assistants and supports to the standard ConvoKit
data format. See :ref:`ConvoKit AI extensions <convokitai-data-format>` in :doc:`Data Format <data_format>`.

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
