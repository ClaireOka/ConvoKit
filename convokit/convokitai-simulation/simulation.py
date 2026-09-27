from pathlib import Path

import deliberate_lab as dl
from convokitai import Corpus
from dl_container import BackendContainer, FIRESTORE_PORT
from local_backend import LocalBackend
from create_simulation import FirebaseBackend, _seed_gemini_api_key, create_simulation


def simulate(
    gemini_api_key: str | None = None,
    sim_yaml: str | Path | dict | None = None,
    *,
    project_id: str | None = None,
    dl_api_key: str | None = None,
    backend: FirebaseBackend | None = None,
    wait_timeout: float | None = None,
) -> Corpus:
    """Run a simulation and return the conversations as a convokitai Corpus.

    Runs on the bundled local backend by default. To run on a deployed
    Deliberate Lab instead, pass its Firebase `project_id` and a `dl_api_key`
    (a Deliberate Lab API key from that deployment's web UI, Settings ->
    API Keys), or a prebuilt `backend`.

    sim_yaml is the simulation YAML: a file path, the YAML text, or the parsed dict.
    wait_timeout is how many seconds to wait for the conversations before keeping
    them as they stand (defaults to the YAML's max_time plus 5 minutes).

    gemini_api_key is required locally. On a deployment it is saved to the
    account that owns dl_api_key before the simulation runs; leave it out to
    use the Gemini key already saved in that deployment's web UI Settings.
    """
    if project_id is not None:
        if backend is not None:
            raise ValueError("pass either project_id or backend, not both")
        if not dl_api_key:
            raise ValueError(
                "dl_api_key is required with project_id; create one in the "
                "deployment's web UI under Settings -> API Keys"
            )
        backend = FirebaseBackend(project_id=project_id, api_key=dl_api_key)

    if backend is not None:
        if gemini_api_key:
            _set_deployed_gemini_api_key(backend, gemini_api_key)
        return create_simulation(backend, sim_yaml, wait_timeout)

    if not gemini_api_key:
        raise ValueError("gemini_api_key is required for the local backend")
    with BackendContainer():
        with LocalBackend(
            ".",
            reuse_running=True,
            firestore_port=FIRESTORE_PORT,
            # the functions can take minutes to load under udocker (Colab)
            startup_timeout=600,
        ) as local:
            _seed_gemini_api_key(local, gemini_api_key)
            return create_simulation(local, sim_yaml, wait_timeout)


def _set_deployed_gemini_api_key(backend: FirebaseBackend, gemini_api_key: str) -> None:
    try:
        backend.client().set_gemini_api_key(gemini_api_key)
    except dl.APIError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            raise RuntimeError(
                f"The deployment at {backend.base_url} has no endpoint for setting "
                "the Gemini key; deploy a TrAuSt backend that includes "
                "PUT /v1/experimenter/api-keys/gemini, or save the key in the web "
                "UI's Settings and call simulate() without gemini_api_key"
            ) from exc
        raise
