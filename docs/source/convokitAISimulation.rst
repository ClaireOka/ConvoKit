ConvoKit AI Simulation
======================

ConvoKit AI Simulation runs conversations between AI agents on ConvoArena, a platform for
online conversation experiments with humans and AI agents built on
`Deliberate Lab <https://github.com/PAIR-code/deliberate-lab>`_, and returns them as a :doc:`ConvoKit AI <convokitai>`
corpus. The conversations can include an AI mediator that speaks in the chat and private assistants
that privately help individual participants.

A simulation is described by a *simulation YAML* that lists the agents, mediators and assistants,
the topics, and who talks to whom. Each pairing of agents and each topic runs as its own Deliberate
Lab experiment. The package adds the agents to the chat, waits for the conversations to finish, and
converts the experiment exports into a corpus.

Simulations run either:

* **locally** (the default): a ConvoArena backend runs on the Firebase emulators in a container
  on your machine, so no Firebase project is needed; or
* **on your own ConvoArena deployment**, using an API key from its web interface. To set up your own ConvoArena deployement check out `<https://docs.google.com/document/d/1pRw3zydkPMAGVHDkWWwNOkJNDq0LVmfuSxHuntwp4yY/edit?tab=t.0>`

Example usage: `simulation demo <https://github.com/CornellNLP/ConvoKit/blob/master/convokit/convokitai-simulation/simulation_demo.ipynb>`_.

Installation
------------

.. important::

   ConvoKit AI Simulation requires **Python 3.12 or above**. This is newer than core ConvoKit, which
   supports Python 3.10 and above, so an environment that works for ConvoKit may be too old for
   this package. Check your version with ``python --version``, and if it is below 3.12, create a new
   environment first, for example::

       python3.12 -m venv .venv
       source .venv/bin/activate

   or with conda::

       conda create -n convokitai python=3.12
       conda activate convokitai

   Installing into an older Python fails with an error that the package
   ``requires a different Python``.

Then, from the repository root, install it together with ConvoKit (which includes ``convokitai``)::

    python -m pip install -e . -e ./convokit/convokitai-simulation

or from GitHub::

    pip install "git+https://github.com/CornellNLP/ConvoKit.git@master#subdirectory=convokit/convokitai-simulation"

The modules are installed as top-level modules (``simulation``, ``create_simulation``,
``export_to_corpus``, ``local_backend``, ``dl_container``), along with ``deliberate_lab``, the
Python client for the ConvoArena REST API (from Deliberate Lab).

Running locally also requires:

* `Docker <https://docs.docker.com/get-docker/>`_. Where Docker isn't available (e.g. Google Colab),
  the backend runs with `udocker <https://github.com/indigo-dc/udocker>`_ instead, which is
  downloaded automatically.
* A `Gemini API key <https://ai.google.dev/gemini-api/docs/api-key>`_ for the agents' model calls.

Quick start
-----------

Run a simulation locally:

.. code-block:: python

    from simulation import simulate

    corpus = simulate(gemini_api_key="...", sim_yaml="simulation.yaml")
    corpus.print_summary_stats()

    for convo in corpus.iter_conversations():
        print(convo.meta["pairing_id"], convo.meta["completed"])
        print(convo.get_transcript(supports=True))

``simulate`` starts the backend container, runs every conversation, and stops the backend when it is
done. The first run downloads the backend image and can take a few minutes to start.

To run on your own ConvoArena deployment instead, create a ConvoArena API key in its web
interface (**Settings → API Keys**) and save your Gemini API key in the same **Settings** page. Then
pass the Firebase project ID instead of the Gemini key:

.. code-block:: python

    corpus = simulate(sim_yaml="simulation.yaml", project_id="my-convoarena", dl_api_key="dlb_live_...")

For a deployment whose Cloud Functions run outside ``us-central1``, build the backend yourself:

.. code-block:: python

    from create_simulation import FirebaseBackend, create_simulation

    backend = FirebaseBackend(project_id="my-convoarena", api_key="dlb_live_...", region="europe-west1")
    corpus = create_simulation(backend, "simulation.yaml")

The simulation YAML
-------------------

``sim_yaml`` can be a path to a YAML file, the YAML text itself, or an already-parsed ``dict``.
For example, a mediator and two agents discussing one topic:

.. code-block:: yaml

    description: Two people discuss a question with the help of a mediator.
    blocks:
      - name: Debate topic
        descriptions:
          - Should cities ban cars from their downtown areas?
    max_utterance: 10
    max_time: 3
    pairings:
      - id: mediated-debate
        members:
          - participant: mediator:mediator
          - participant: skeptic
            assistant: tone-checker
          - participant: enthusiast
    definitions:
      agents:
        skeptic:
          content:
            persona: {id: skeptic, name: Skeptic, avatar: "🦉", character: cautious retiree}
            model: {apiType: GEMINI, modelName: gemini-3-flash-preview}
            chatSettings:
              promptMap:
                message:
                  order: 1
                  prompt:
                    - {type: TEXT, text: "You are a "}
                    - {type: CHARACTER_CONTEXT}
                    - {type: TEXT, text: " discussing: {topic_statement}"}
                    - {type: CONTEXT, context: current}
        enthusiast: ...
      mediators:
        mediator: ...
      assistants:
        tone-checker: ...

See the `simulation demo <https://github.com/CornellNLP/ConvoKit/blob/master/convokit/convokitai-simulation/simulation_demo.ipynb>`_
for a complete example with a mediator and a private assistant.

Top-level fields
^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Field
     - Description
   * - ``description``
     - Text shown with the chat.
   * - ``blocks``
     - Text blocks shown with the chat, each with a ``name`` and a list of ``descriptions``. The first
       block fills the ``{topic_name}`` and ``{topic_statement}`` placeholders in prompts.
   * - ``max_utterance``
     - The chat ends after this many messages.
   * - ``max_time``
     - The chat ends after this many minutes.
   * - ``pairings``
     - The conversations to run. Each pairing has an ``id`` and a list of ``members``. A member's
       ``participant`` is ``mediator:<key>`` for a mediator or ``<key>`` for an agent, with an
       optional ``assistant`` (a key in ``definitions.assistants``) and ``role``.
   * - ``definitions``
     - The ``agents``, ``mediators`` and ``assistants`` that pairings refer to, keyed by name. Each
       entry's ``content`` has a ``persona`` (``id``, ``name``, ``avatar``, and for agents an optional
       ``character``), a ``model`` (``apiType``, ``modelName``), ``generation`` settings and a prompt.

One conversation runs for each pairing and each combination of block descriptions. For example,
two pairings and a block with three descriptions run six conversations.

Prompt items
^^^^^^^^^^^^

Prompts are lists of items, which are translated into ConvoArena prompt items:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Item ``type``
     - Inserts
   * - ``TEXT``
     - ``text``, with ``{topic_name}``, ``{topic_statement}``, ``{post_title}`` and
       ``{post_description}`` filled in.
   * - ``CONTEXT``
     - The chat so far (``context: current``), the stages before it (``before``), or both (``all``).
   * - ``CHARACTER_CONTEXT``
     - The agent's ``persona.character``.
   * - ``PARTICIPANT_ROLE``
     - The member's ``role`` from the pairing.
   * - ``PROFILE_INFO``, ``PARTICIPANT_INFO``, ``PROFILE_CONTEXT``
     - Information about the agent itself, or (for an assistant) the participant it helps.
   * - ``PARTICIPANT_CHAT_INPUT``
     - The message the participant is drafting (for assistants).
   * - ``INITIALIZATION_CONTEXT`` (or ``PRELOADED_CONTEXT``)
     - The output of the mediator's ``initialization_context_prompt``, run once before the chat.
   * - ``POST_TITLE``, ``POST_DESCRIPTION``, ``ARTICLE_PAGE``
     - The post title and description (``post_title`` / ``post_description``, defaulting to the
       first topic and the ``description``).
   * - ``RULE``
     - A rule of the r/ChangeMyView subreddit, given by ``rule`` (``A``–``E`` or ``1``–``5``).

An agent prompt without any ``TEXT`` instructions gets a default instruction to reply in character
with a short message, so that the agent always has something to do.

The resulting corpus
--------------------

The corpus has one conversation per pairing and topic combination:

* Agents and mediators are AI speakers. Mediators have ``ai_meta["role"] == "public assistant"`` and
  agents have ``"participant"``. ``ai_meta["config"]`` holds the model and prompt, and the full
  ConvoArena agent template under ``"deliberate_lab"``.
* ``convo.alias`` maps speaker IDs to the persona names used in the chat.
* Utterances keep the model's stated reason for the message in ``ai_meta["explanation"]`` when the
  agent's structured output includes one.
* Private assistants and their supports are stored as
  :class:`~convokitai.private_assistant.PrivateAssistant` and :class:`~convokitai.support.Support`
  objects. Each support keeps the participant's draft at the time.
* ``convo.meta`` has the ``pairing_id``, whether the conversation ``completed`` before the timeout,
  the topic ``blocks`` shown, and the ConvoArena ``experiment_id``, ``experiment_name`` and
  ``cohort_name``.
* ``corpus.ai_meta["simulation_config"]`` holds the parsed simulation YAML.

Because mediators are marked as ``"public assistant"`` and names are in ``convo.alias``, the corpus
works directly with :doc:`ConvoKit AI Evaluation <convokitAIEvaluation>`.

API reference
-------------

Running simulations
^^^^^^^^^^^^^^^^^^^

.. automodule:: simulation
    :members:

.. automodule:: create_simulation
    :members: create_simulation, FirebaseBackend, load_simulation_config

Converting ConvoArena exports
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: export_to_corpus
    :members:

Local backend
^^^^^^^^^^^^^

.. automodule:: dl_container
    :members: BackendContainer, BackendContainerError

.. automodule:: local_backend
    :members: LocalBackend, LocalBackendError
