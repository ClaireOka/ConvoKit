## Assistant

| Field      | Type      | Description                                  |
|------------|-----------|----------------------------------------------|
| `id`       | `string`  | Unique ID for the assistant.                 |
| `config` | `dict` | Configuration for this assistant, including the `prompt` that generates a support message from it (required by `Assistant.generate`). See the example below |
| `Support[]` | `list` | List of support objects |
| `speakers` | `list` | List of speakers that can see this Assistant's supports |
| `conversation_id` | `string` | id of conversation |
| `meta` | `dict` | Other metadata, e.g. outputs of transformers run on the assistant (such as `LLMPromptTransformer`) |

config example
```python
{
    'prompt': 'You help goose phrase their replies politely.',
    'model':'gemini-2.5-flash',
    'temperature': 0.7
}
```

## Support

| Field | Type | Description |
|---|---|---|
| `id` | `string` | Unique ID for the support. |
| `text` | `string` | Output of the support |
| `reply_to` | utt_id | The utterance the assisted speaker is replying  |
| `draft` | `string` | The draft of the post that the assisted speaker has written so far. empty string means draft is empty  |
| `assistant_id` | `string` | id of assistant |
| `timestamp` | `string` | The timestamp the message was sent  |
| `meta` | `dict` | Other metadata, e.g. outputs of transformers run on the support (such as `LLMPromptTransformer`) |

A support can be stored in several places of a conversation (its assistant's `Support[]`, the conversation's `Support[]`, and the `Support[]` of the utterance it replies to). On load, these copies are linked into one object, so changes to one of them (e.g. to its `meta`) show up in all of them.

## Speaker
| Field     | Type     | Description                           |
| --------- | -------- | ------------------------------------- |
| `id`      | `string` | Unique ID for the speaker.            |
| `is_ai`   | `bool`   | Whether this speaker is known to be an LLM agent. |
| `ai_meta` | `dict`   | Metadata for ConvoKitAI. (empty if is_ai is false)              |

`ai_meta`

| Field    | Type     | Description                                                         |
| -------- | -------- | ------------------------------------------------------------------- |
| `config` | `dict`   | Config that generates a message for this speaker (used by `Speaker.generate`): `prompt` (required), `model`, optional `provider` (`gemini` / `gpt` / `local`, inferred from `model`) and `temperature`. |
| `role`   | `string` | Speaker's role in the conversation, e.g. `participant`, `mediator`. |


## Conversation
| Field     | Type     | Description                     |
| --------- | -------- | ------------------------------- |
| `id`      | `string` | Unique ID for the conversation. |
| `ai_meta` | `dict`   | Metadata for ConvoKitAI.        |

`ai_meta`
| Field        | Type   | Description                                                                                          |
| ------------ | ------ | ---------------------------------------------------------------------------------------------------- |
| `alias`      | `dict` | Names speakers are referred to in the conversation, e.g. `bear`, `goose`, etc., keyed by speaker ID. |
| `assistants` | `list` | List of assistants                             |
| `Support[]` | `list` | List of supports created from all assistants in this conversation                         |

## Utterance
| Field             | Type     | Description                                                                                                                             |
| ----------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `id`              | `string` | Unique ID for the utterance.                                                                                                            |
| `conversation_id` | `string` | ID of the conversation the utterance belongs to.                                                                                        |
| `reply_to`        | `string` | Index of the utterance to which this utterance replies. Defaults to the previous utterance.                                             |
| `timestamp`       | `int`    | Time of the utterance.                                                                                                                  |
| `text`            | `string` | Textual content of the utterance.                                                                                                       |
| `ai_meta`         | `dict`   | Additional outputs from the speaker LLM call for this message. Empty if there are no additional outputs or the speaker is not an agent. |

`ai_meta`
| Field             | Type     | Description                                                                                                                             |
| ----------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `config`              | `dict` | config used to generate message|
| `Support[]`              | `list` | list of supports attatched to this utterance |

## Corpus
| Field     | Type     | Description                                    |
| --------- | -------- | ---------------------------------------------- |
| `has_ai`  | `bool`   | `True` if there are AI speakers in the corpus. |
| `ai_meta` | `dict`   | Metadata for ConvoKitAI.                       |

`ai_meta`
| Field        | Type   | Description                                                                                                                                     |
| ------------ | ------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |

