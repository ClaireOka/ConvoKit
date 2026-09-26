from dl_container import BackendContainer, FIRESTORE_PORT
from local_backend import LocalBackend
from create_simulation import _seed_gemini_api_key, create_simulation
import yaml

def simulate(gemini_api_key, sim_yaml: yaml):
    with BackendContainer():
        with LocalBackend(".", reuse_running=True, firestore_port=FIRESTORE_PORT) as backend:
            _seed_gemini_api_key(backend, gemini_api_key)
            return create_simulation(backend, sim_yaml)