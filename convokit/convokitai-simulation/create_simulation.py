"""run_conversation.py — spin up the local Deliberate Lab emulator via
LocalBackend and run a two-agent chat conversation end to end, then print
the transcript.

Adapted from scripts/startup.py's create_demo(), trimmed to just a chat
stage between two agent participants (no humans, no survey/mediator), and
pointed at the emulator through LocalBackend instead of a hardcoded prod URL.

Requirements:
    - A deliberate-lab checkout built per local_backend.py's docstring.
    - The functions emulator needs a real LLM key configured (e.g. GEMINI_API_KEY
      in the checkout's .env / functions config) — otherwise the agents will be
      created but every response call will fail, since dl.ApiKeyType.GEMINI below
      is not a mock.

Usage:
    python scripts/run_conversation.py /path/to/deliberate-lab
"""

from __future__ import annotations
import yaml
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from dataclasses import dataclass

import requests
from dotenv import dotenv_values

import deliberate_lab as dl
from local_backend import LocalBackend

from dataclasses import dataclass


@dataclass
class FirebaseBackend:
    """A user's own deployed Deliberate Lab. api_key is a Deliberate Lab API key
    created in that deployment's web UI (Settings -> API Keys)."""

    project_id: str
    api_key: str
    region: str = "us-central1"

    @property
    def base_url(self) -> str:
        return f"https://{self.region}-{self.project_id}.cloudfunctions.net/api/v1"

    def client(self) -> dl.Client:
        return dl.Client(base_url=self.base_url, api_key=self.api_key)


TOPIC = "Should pineapple be allowed on pizza?"


def _structured_output_config() -> dl.AgentParticipantStructuredOutputConfig:
    return dl.AgentParticipantStructuredOutputConfig(
        enabled=True,
        type=dl.StructuredOutputType.JSON_SCHEMA,
        appendToPrompt=False,
        shouldRespondField="shouldRespond",
        messageField="response",
        explanationField="explanation",
        readyToEndField="readyToEndChat",
        schema=dl.StructuredOutputSchema(
            type=dl.StructuredOutputDataType.OBJECT,
            properties=[
                dl.StructuredOutputSchemaProperty(
                    name="explanation",
                    schema=dl.StructuredOutputSchema(
                        type=dl.StructuredOutputDataType.STRING,
                        description="1-2 sentences on why you are responding or staying silent.",
                    ),
                ),
                dl.StructuredOutputSchemaProperty(
                    name="shouldRespond",
                    schema=dl.StructuredOutputSchema(
                        type=dl.StructuredOutputDataType.BOOLEAN,
                        description="True to send a message this turn, false to stay silent.",
                    ),
                ),
                dl.StructuredOutputSchemaProperty(
                    name="response",
                    schema=dl.StructuredOutputSchema(
                        type=dl.StructuredOutputDataType.STRING,
                        description="Your chat message (empty string if staying silent).",
                    ),
                ),
                dl.StructuredOutputSchemaProperty(
                    name="readyToEndChat",
                    schema=dl.StructuredOutputSchema(
                        type=dl.StructuredOutputDataType.BOOLEAN,
                        description="Whether you are ready to end the conversation.",
                    ),
                ),
            ],
        ),
    )


def _agent_template(agent_id: str, name: str, stance: str) -> dl.AgentParticipantTemplate:
    persona_prompt = f"""
You are {name}, a participant in a live discussion about "{TOPIC}".
Your stance: {stance}

Respond ONLY with JSON matching this schema, no markdown, no prose outside the JSON:
{{"explanation": "...", "shouldRespond": true, "response": "...", "readyToEndChat": false}}

Keep "response" under 20 words when you do respond. Stay in character.
"""
    chat_prompt = dl.ChatPromptConfig(
        id="discussion",
        type=dl.ChatStageType.chat,
        includeScaffoldingInPrompt=True,
        prompt={
            "default": [
                dl.TextPromptItem(type="TEXT", text=persona_prompt),
                dl.ProfileInfoPromptItem(type="PROFILE_INFO"),
                dl.StageContextPromptItem(
                    type="STAGE_CONTEXT",
                    stageId="discussion",
                    includePrimaryText=True,
                    includeInfoText=False,
                    includeHelpText=False,
                    includeStageDisplay=True,
                    includeParticipantAnswers=True,
                ),
            ]
        },
        order={1: ["default"]},
        addTo={},
        structuredOutputConfig=_structured_output_config(),
        generationConfig=dl.ModelGenerationConfig(
            temperature=0.7,
            reasoningLevel=dl.ReasoningLevel.off,
            includeReasoning=False,
        ),
        chatSettings=dl.AgentChatSettings(
            minMessagesBeforeResponding=0,
            canSelfTriggerCalls=False,
            initialMessage="",
            wordsPerMinute=0,
            concedeStrength=0,
        ),
        numRetries=2,
    )
    return dl.AgentParticipantTemplate(
        persona=dl.ParticipantPersona(
            id=agent_id,
            name=name,
            defaultModelSettings=dl.AgentModelSettings(
                apiType=dl.ApiKeyType.GEMINI,
                modelName="gemini-3-flash-preview",
            ),
        ),
        promptMap={"discussion": chat_prompt},
    )


def _seed_gemini_api_key(backend: LocalBackend, gemini_api_key: str) -> None:
    """Write the Gemini key into experimenterData/{email} in the Firestore
    emulator, matching scripts/seed-api-key.mjs (which run_locally.sh calls
    but LocalBackend never does). createAgentChatMessageFromPrompt reads this
    via getExperimenterDataFromExperiment and silently no-ops without it —
    no error, no chat message, which is why agents go quiet with zero logs.
    """
    url = (
        f"http://127.0.0.1:{backend.firestore_port}/v1/projects/"
        f"{backend.project_id}/databases/(default)/documents/experimenterData"
    )
    document = {
        "fields": {
            "id": {"stringValue": backend.experimenter_email},
            "email": {"stringValue": backend.experimenter_email},
            "apiKeys": {
                "mapValue": {
                    "fields": {
                        "geminiApiKey": {"stringValue": gemini_api_key},
                        "ollamaApiKey": {
                            "mapValue": {
                                "fields": {
                                    "url": {"stringValue": ""},
                                    "apiKey": {"stringValue": ""},
                                }
                            }
                        },
                    }
                }
            },
            "viewedExperiments": {"arrayValue": {"values": []}},
            "showAlphaFeatures": {"booleanValue": False},
        }
    }
    resp = requests.post(
        url,
        params={"documentId": backend.experimenter_email},
        headers={"Authorization": "Bearer owner", "Content-Type": "application/json"},
        json=document,
        timeout=30,
    )
    if not resp.ok:
        raise RuntimeError(f"seeding Gemini API key failed: {resp.status_code} {resp.text}")


def _add_agent_to_cohort(
    backend: LocalBackend | FirebaseBackend,
    experiment_id: str,
    cohort_id: str,
    agent: dl.AgentParticipantTemplate,
) -> dict:
    """Spawn a participant instance of an agent template into a cohort.

    Templates passed to create_simulation register the persona/prompt on the
    experiment; they don't join a cohort until createParticipant is called.
    agentConfig.modelSettings is required by CreateParticipantData
    (utils/src/participant.validation.ts) even though the template already
    carries defaultModelSettings.
    """
    url = (
        f"http://127.0.0.1:{backend.functions_port}/{backend.project_id}"
        f"/{backend.region}/createParticipant"
    )
    model_settings = agent.persona.defaultModelSettings
    resp = requests.post(
        url,
        json={
            "data": {
                "experimentId": experiment_id,
                "cohortId": cohort_id,
                "isAnonymous": True,
                "agentConfig": {
                    "agentId": agent.persona.id,
                    "promptContext": "",
                    "modelSettings": {
                        "apiType": model_settings.apiType,
                        "modelName": model_settings.modelName,
                    },
                },
            }
        },
        timeout=60,
    )
    if not resp.ok:
        raise RuntimeError(
            f"createParticipant failed: {resp.status_code} {resp.text}"
        )
    body = resp.json()
    if "error" in body:
        raise RuntimeError(f"createParticipant error: {body['error']}")
    return body.get("result", body)


def _heartbeat(log_path: Path, stop: threading.Event, interval: float = 5.0) -> None:
    start = time.monotonic()
    while not stop.wait(interval):
        elapsed = int(time.monotonic() - start)
        size = log_path.stat().st_size if log_path.exists() else 0
        print(f"  ... still waiting on emulators ({elapsed}s elapsed, log is {size} bytes)")


def create_simulation(backend: LocalBackend | FirebaseBackend, sim_yaml: str) -> dict:
    """Create a two-agent chat experiment on the running backend, add both
    agents to its cohort, and block until the conversation finishes.
    Returns the completed experiment export.
    """
    client = backend.client()

    stage = dl.ChatStageConfig(
        id="discussion",
        kind="chat",
        name="Discussion",
        descriptions={
            "primaryText": f'Discuss: "{TOPIC}"',
            "infoText": "",
            "helpText": "",
        },
        progress={
            "minParticipants": 2,
            "waitForAllParticipants": False,
            "showParticipantProgress": True,
        },
        timeLimitInMinutes=2,
        numUtterances=5,
        discussions=[dl.DefaultChatDiscussion(id="main", description=TOPIC)],
    )

    agents = [
        _agent_template("agent-pro", "Pat", "You love pineapple on pizza."),
        _agent_template("agent-anti", "Alex", "You think pineapple ruins pizza."),
    ]

    result = client.create_simulation(
        name="simulation test",
        stages=[stage],
        agent_participants=agents,
        num_cohorts=1,
        cohort_names=["cohort"],
        cohort_participant_config=[
            dl.CohortParticipantConfig(
                minParticipantsPerCohort=2,
                maxParticipantsPerCohort=2,
                includeAllParticipantsInCohortCount=True,
                botProtection=False,
            )
        ],
    )

    experiment_id = result["experiment"]["experiment"]["id"]
    cohort = result["cohorts"][0]
    cohort_id = cohort["cohort"]["id"] if "cohort" in cohort else cohort["id"]
    print(f"created experiment {experiment_id}, cohort {cohort_id}")

    for agent in agents:
        print(f"adding {agent.persona.id} ({agent.persona.name}) to cohort...")
        _add_agent_to_cohort(backend, experiment_id, cohort_id, agent)
        print(f"  {agent.persona.id} joined")
    print("all agents joined, waiting for the conversation to finish...")

    poll_start = time.monotonic()
    while True:
        try:
            return client.get_completed_experiment_data(experiment_id)
        except RuntimeError:
            elapsed = int(time.monotonic() - poll_start)
            print(f"  ... still running ({elapsed}s elapsed)")
            time.sleep(5)


def main(repo_root: str, gemini_api_key: str | None = None) -> None:
    log_path = Path(tempfile.gettempdir()) / f"dl-run-conversation-{int(time.time())}.log"
    print("starting emulators (first boot can take a couple minutes) — tail progress with:")
    print(f"  tail -f {log_path}")

    stop_heartbeat = threading.Event()
    heartbeat = threading.Thread(
        target=_heartbeat, args=(log_path, stop_heartbeat), daemon=True
    )
    heartbeat.start()
    backend_cm = LocalBackend(repo_root, log_path=log_path)
    try:
        backend = backend_cm.__enter__()
    finally:
        stop_heartbeat.set()
        heartbeat.join(timeout=1)

    try:
        print(f"backend up at {backend.base_url}")

        if gemini_api_key:
            _seed_gemini_api_key(backend, gemini_api_key)
            print("seeded Gemini API key into experimenterData")
        else:
            print(
                "WARNING: no GEMINI_API_KEY found — agents will join but "
                "every chat turn will silently fail with no messages sent"
            )

        export = create_simulation(backend, None)

        print("done. full export:")
        print(json.dumps(export, indent=2))
    finally:
        backend_cm.__exit__(*sys.exc_info())
        print("backend stopped")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")