from pathlib import Path

from dl_container import BackendContainer, FIRESTORE_PORT
from local_backend import LocalBackend
from create_simulation import FirebaseBackend, _seed_gemini_api_key, create_simulation


def simulate(
    gemini_api_key: str | None = None,
    sim_yaml: str | Path | dict | None = None,
    *,
    backend: FirebaseBackend | None = None,
) -> dict:
    """Run a simulation on the bundled local backend, or on your own deployed
    Deliberate Lab if `backend` is given.

    gemini_api_key is required locally. A deployed backend uses the Gemini key
    saved in its web UI's Settings by the account that owns the API key.
    """
    if backend is not None:
        if gemini_api_key:
            raise ValueError(
                "gemini_api_key is only used with the local backend; on your own "
                "deployment, save the Gemini key in the web UI's Settings instead"
            )
        return create_simulation(backend, sim_yaml)

    if not gemini_api_key:
        raise ValueError("gemini_api_key is required for the local backend")
    with BackendContainer():
        with LocalBackend(".", reuse_running=True, firestore_port=FIRESTORE_PORT) as local:
            _seed_gemini_api_key(local, gemini_api_key)
            return create_simulation(local, sim_yaml)
