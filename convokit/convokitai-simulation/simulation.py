from dl_container import BackendContainer, FIRESTORE_PORT
from local_backend import LocalBackend
from create_simulation import _seed_gemini_api_key, create_simulation


def simulate(gemini_api_key: str, sim_yaml: str | None = None) -> dict:
    """Run a simulation. sim_yaml (not used yet) will describe a custom simulation."""
    with BackendContainer():
        with LocalBackend(".", reuse_running=True, firestore_port=FIRESTORE_PORT) as backend:
            _seed_gemini_api_key(backend, gemini_api_key)
            return create_simulation(backend, sim_yaml)
