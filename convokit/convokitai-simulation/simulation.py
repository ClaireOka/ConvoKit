from pathlib import Path

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

    gemini_api_key is required locally. A deployed backend uses the Gemini key
    saved in its web UI's Settings by the account that owns the API key.
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
            raise ValueError(
                "gemini_api_key is only used with the local backend; on your own "
                "deployment, save the Gemini key in the web UI's Settings instead"
            )
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

