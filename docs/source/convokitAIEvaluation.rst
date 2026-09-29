ConvoKit AI Evaluation
======================

Metrics for AI-mediated conversations in the ConvoKit AI format: conversational
information gain (CIG), claim-level consensus, and mediator redirection. Requires
``convokitai`` and an OpenAI key (``GPT_API_KEY``); redirection also needs a Hugging Face token.

Example usage: `cig_demo.ipynb <convokit/convokitai_evaluation/examples/cig_demo.ipynb>`_,
`claims_demo.ipynb <convokit/convokitai_evaluation/examples/claims_demo.ipynb>`_,
`redirection_demo.ipynb <convokit/convokitai_evaluation/examples/redirection_demo.ipynb>`_.

CIG
---

.. automodule:: convokitai_evaluation.cig
    :members:

Claim tracking
--------------

.. automodule:: convokitai_evaluation.claims
    :members:

Mediator redirection
--------------------

.. automodule:: convokitai_evaluation.redirection
    :members:

LLM client
----------

.. automodule:: convokitai_evaluation.llm
    :members:
