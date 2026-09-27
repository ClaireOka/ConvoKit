"""export_to_corpus.py — convert Deliberate Lab experiment exports (the JSON returned by
Client.export_experiment) into a convokitai Corpus.

Each cohort becomes a Conversation. Participant and mediator chat messages become
Utterances (system messages are skipped), and each participant's private assistant
messages become Supports of that conversation's Assistants.

Usage:
    import json
    from export_to_corpus import export_to_corpus

    with open("simulation-exp-....json") as f:
        corpus = export_to_corpus(json.load(f))
"""

from __future__ import annotations

from typing import Iterable

from convokitai import Assistant, Corpus, Speaker, Support, Utterance

# what an assistant "says" when it decides not to intervene
SILENT_SUPPORT_TEXT = "Nothing further to add at this point in the conversation."

# Deliberate Lab apiType -> convokit.genai provider
PROVIDERS = {"GEMINI": "gemini", "OPENAI": "gpt", "OLLAMA": "local"}


def _timestamp_ms(timestamp) -> int | None:
    """Firestore timestamp ({seconds, nanoseconds}, or _seconds/_nanoseconds) -> milliseconds."""
    if not isinstance(timestamp, dict):
        return timestamp
    seconds = timestamp.get("seconds", timestamp.get("_seconds", 0))
    nanoseconds = timestamp.get("nanoseconds", timestamp.get("_nanoseconds", 0))
    return int(seconds) * 1000 + int(nanoseconds) // 1_000_000


def _prompt_text(prompt) -> str:
    """Join the TEXT items of a prompt (a list of items, or a dict of lists) into one string.
    Context items are dropped, since convokitai adds the transcript itself when generating."""
    if isinstance(prompt, dict):
        prompt = [item for items in prompt.values() for item in items]
    return "".join(item.get("text", "") for item in prompt or [] if item.get("type") == "TEXT").strip()


def _generation_config(template: dict | None, stage_id: str, model_settings: dict | None = None) -> dict:
    """Build a convokitai generation config from a Deliberate Lab agent template. The full
    template is kept under "deliberate_lab" so the simulation can be recreated."""
    if not template:
        return {}
    persona = template.get("persona", {})
    prompt_config = template.get("promptMap", {}).get(stage_id, {})
    model_settings = model_settings or persona.get("defaultModelSettings") or {}

    config = {"prompt": _prompt_text(prompt_config.get("prompt"))}
    if model_settings.get("modelName"):
        config["model"] = model_settings["modelName"]
    if model_settings.get("apiType") in PROVIDERS:
        config["provider"] = PROVIDERS[model_settings["apiType"]]
    temperature = (prompt_config.get("generationConfig") or {}).get("temperature")
    if temperature is not None:
        config["temperature"] = temperature
    config["deliberate_lab"] = template
    return config


def _without_template(config: dict) -> dict:
    return {k: v for k, v in config.items() if k != "deliberate_lab"}


def _chat_stage_ids(export: dict) -> list[str]:
    stage_map = export.get("stageMap", {})
    stage_ids = export.get("experiment", {}).get("stageIds") or list(stage_map)
    return [sid for sid in stage_ids if stage_map.get(sid, {}).get("kind") == "chat"]


def _add_export(
    export: dict,
    utterances: list[Utterance],
    convo_ai_meta: dict[str, dict],
    convo_meta: dict[str, dict],
    keep_silent_supports: bool,
) -> None:
    experiment = export.get("experiment", {})
    participant_map = export.get("participantMap", {})
    mediator_map = export.get("agentMediatorMap", {})
    agent_map = export.get("agentParticipantMap", {})
    assistant_map = export.get("agentAssistantMap", {})
    chat_stage_ids = _chat_stage_ids(export)
    # configs are per stage; with several chat stages, use the first one's
    config_stage = chat_stage_ids[0] if chat_stage_ids else None

    for cohort_id, cohort_data in export.get("cohortMap", {}).items():
        speakers: dict[str, Speaker] = {}
        alias: dict[str, str] = {}
        prev_id = None
        utt_ids = set()

        messages = [
            (stage_id, message)
            for stage_id in chat_stage_ids
            for message in cohort_data.get("chatMap", {}).get(stage_id, [])
            if message.get("type") in ("participant", "mediator")
        ]
        messages.sort(key=lambda m: _timestamp_ms(m[1].get("timestamp")) or 0)

        for stage_id, message in messages:
            sender_id = message["senderId"]
            if sender_id not in speakers:
                if message["type"] == "mediator":
                    config = _generation_config(mediator_map.get(message.get("agentId")), config_stage)
                    speakers[sender_id] = Speaker(
                        id=sender_id, is_ai=True, ai_meta={"role": "mediator", "config": config}
                    )
                else:
                    profile = participant_map.get(sender_id, {}).get("profile", {})
                    agent_config = profile.get("agentConfig")
                    ai_meta = {"role": "participant"}
                    if agent_config:
                        ai_meta["config"] = _generation_config(
                            agent_map.get(agent_config.get("agentId")),
                            config_stage,
                            agent_config.get("modelSettings"),
                        )
                    speakers[sender_id] = Speaker(
                        id=sender_id, is_ai=agent_config is not None, ai_meta=ai_meta
                    )
                alias[sender_id] = (message.get("profile") or {}).get("name") or sender_id

            speaker = speakers[sender_id]
            ai_meta = {}
            if speaker.ai_meta.get("config"):
                ai_meta["config"] = _without_template(speaker.ai_meta["config"])
            if message.get("explanation"):
                ai_meta["explanation"] = message["explanation"]

            utterances.append(
                Utterance(
                    id=message["id"],
                    speaker=speaker,
                    conversation_id=cohort_id,
                    reply_to=prev_id,
                    timestamp=_timestamp_ms(message.get("timestamp")),
                    text=message.get("message", ""),
                    meta={"stage_id": stage_id},
                    ai_meta=ai_meta,
                )
            )
            prev_id = message["id"]
            utt_ids.add(message["id"])

        # assistants, and the supports each participant received from theirs
        assistants: dict[str, Assistant] = {}
        for participant in participant_map.values():
            profile = participant.get("profile", {})
            if profile.get("currentCohortId") != cohort_id:
                continue
            assistant_id = (profile.get("agentConfig") or {}).get("assistantId")
            if assistant_id:
                assistant = assistants.setdefault(
                    assistant_id,
                    Assistant(
                        id=assistant_id,
                        config=_generation_config(assistant_map.get(assistant_id), config_stage),
                        conversation_id=cohort_id,
                    ),
                )
                assistant.speakers.append(profile["publicId"])
                alias.setdefault(profile["publicId"], profile.get("name") or profile["publicId"])

            for stage_id in chat_stage_ids:
                for record in participant.get("assistantChatMap", {}).get(stage_id, []):
                    if record.get("isError"):
                        continue
                    if not keep_silent_supports and record.get("message") == SILENT_SUPPORT_TEXT:
                        continue
                    record_assistant_id = record.get("agentId") or assistant_id
                    assistant = assistants.setdefault(
                        record_assistant_id,
                        Assistant(
                            id=record_assistant_id,
                            config=_generation_config(assistant_map.get(record_assistant_id), config_stage),
                            speakers=[profile["publicId"]],
                            conversation_id=cohort_id,
                        ),
                    )
                    reply_to = record.get("lastChatMessageId")
                    assistant.supports.append(
                        Support(
                            id=record["id"],
                            text=record.get("message", ""),
                            reply_to=reply_to if reply_to in utt_ids else None,
                            draft=record.get("chatInput", ""),
                            assistant_id=record_assistant_id,
                            timestamp=_timestamp_ms(record.get("timestamp")),
                        )
                    )

        for assistant in assistants.values():
            assistant.supports.sort(key=lambda s: s.timestamp or 0)

        chat_stage = export.get("stageMap", {}).get(config_stage, {}) if config_stage else {}
        descriptions = chat_stage.get("descriptions", {})
        convo_meta[cohort_id] = {
            "experiment_id": experiment.get("id"),
            "experiment_name": experiment.get("metadata", {}).get("name"),
            "cohort_name": cohort_data.get("cohort", {}).get("metadata", {}).get("name"),
            "description": descriptions.get("primaryText", ""),
            "blocks": descriptions.get("blocks", []),
        }
        convo_ai_meta[cohort_id] = {"alias": alias, "assistants": list(assistants.values())}


def export_to_corpus(exports: dict | Iterable[dict], keep_silent_supports: bool = False) -> Corpus:
    """
    Convert one or more Deliberate Lab experiment exports into a convokitai Corpus, with one
    Conversation per cohort.

    :param exports: an experiment export, or a list of them
    :param keep_silent_supports: whether to keep the assistant messages sent when an assistant
        decided not to intervene (SILENT_SUPPORT_TEXT)
    :return: the Corpus
    """
    if isinstance(exports, dict):
        exports = [exports]

    utterances: list[Utterance] = []
    convo_ai_meta: dict[str, dict] = {}
    convo_meta: dict[str, dict] = {}
    for export in exports:
        _add_export(export, utterances, convo_ai_meta, convo_meta, keep_silent_supports)

    if not utterances:
        raise ValueError("the exports contain no chat messages to build a corpus from")

    corpus = Corpus(utterances=utterances)
    corpus.has_ai = any(speaker.is_ai for speaker in corpus.iter_speakers())
    for convo in corpus.iter_conversations():
        for key, value in convo_meta.get(convo.id, {}).items():
            convo.meta[key] = value
        convo.ai_meta = convo_ai_meta.get(convo.id, {})
    return corpus
