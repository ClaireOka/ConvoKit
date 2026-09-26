"""
local_backend.py — run the Deliberate Lab TypeScript backend from Python.

`LocalBackend` supervises the Firebase emulator suite (a Node/Java process
tree) as an explicit, scoped service dependency, and mints an API key so the
`deliberate_lab` Python client can talk to it.

Context-manager only, by design. There is no autostart in ``__init__``, no
``__del__`` hook and no ``weakref.finalize``: object lifetime is a poor proxy
for "how long do I need this service", and tying a process tree to the garbage
collector strands ports when a reference drops early. ``atexit`` is registered
purely as a backstop for the case where a script dies without unwinding.

    from local_backend import LocalBackend

    with LocalBackend("~/src/deliberate-lab") as backend:
        client = backend.client()
        print(client.health_check())
        result = client.create_experiment(name="Smoke test")

Requirements: Python 3.12+, Node with `npx`, a `npm ci`-installed and built
checkout (`npm run build:utils && npm run build:functions`), and Java for the
Firestore/Auth emulators. Only the standard library is used here; the
`deliberate_lab` client itself is imported lazily by `client()`.

Note that the emulator ports are hardcoded in the repo's firebase.json, so at
most one backend can run per machine. This class refuses to start when a port
is already bound unless `reuse_running=True`, in which case it attaches and
never kills a process it did not spawn.
"""

from __future__ import annotations

import atexit
import base64
import hashlib
import json
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

__all__ = ["LocalBackend", "LocalBackendError"]

_IS_WINDOWS = sys.platform == "win32"

# Must match functions/src/dl_api/dl_api_key.utils.ts, which calls Node's
# crypto.scrypt with its default cost parameters and passes the salt as a hex
# *string* (so the UTF-8 bytes of the hex digits are the salt, not the decoded
# bytes). Verified byte-for-byte against Node.
_SCRYPT_N = 16384
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_DKLEN = 64
_KEY_PREFIX = "dlb_live_"


class LocalBackendError(RuntimeError):
    """Raised when the backend cannot be started, reached, or authenticated."""


class LocalBackend:
    """Supervises the Deliberate Lab Firebase emulators for the duration of a
    ``with`` block and exposes the base URL and API key a client needs."""

    def __init__(
        self,
        repo_root: str | os.PathLike[str],
        *,
        project_id: str = "demo-project-id",
        experimenter_email: str = "experimenter@google.com",
        emulators: Sequence[str] = ("functions", "firestore", "auth"),
        import_dir: str | os.PathLike[str] | None = "emulator_test_config",
        export_on_exit: str | os.PathLike[str] | None = None,
        region: str = "us-central1",
        functions_port: int = 5001,
        firestore_port: int = 8080,
        auth_port: int = 9099,
        api_key: str | None = None,
        startup_timeout: float = 180.0,
        shutdown_timeout: float = 20.0,
        log_path: str | os.PathLike[str] | None = None,
        reuse_running: bool = False,
    ) -> None:
        """
        Args:
            repo_root: Checkout containing firebase.json. Unused when
                attaching with ``reuse_running=True``.
            project_id: Passed explicitly as ``--project`` so the URL path and
                the emulator always agree, sidestepping the mismatch between
                .firebaserc.example ("demo-project-id") and the DEV_URL baked
                into the Python client ("demo-deliberate-lab").
            experimenter_email: Identity the minted key belongs to. Experiments
                are filtered by ``metadata.creator == experimenterId``, so this
                must match the account you log into the web UI with if you want
                to see API-created experiments there. The seeded
                emulator_test_config account is experimenter@google.com.
            emulators: Which emulators to start. The REST API needs functions
                and firestore; auth is only needed if you also use the UI.
            import_dir: Emulator data to ``--import``, relative to repo_root.
                Pass None to start from empty state.
            export_on_exit: Directory to ``--export-on-exit`` into, which makes
                minted keys and created experiments survive a restart.
            api_key: Use an existing key instead of minting one. Falls back to
                $DL_API_KEY, then to minting.
            startup_timeout: Emulator boot is slow; the repo's own
                run_locally.sh allows 120s per port.
            log_path: Where to tee child output. Defaults to a temp file whose
                tail is included in exceptions.
            reuse_running: Attach to an already-running emulator suite instead
                of failing on a port conflict. Nothing it did not start is
                ever killed.
        """
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.project_id = project_id
        self.experimenter_email = experimenter_email.lower()
        self.emulators = tuple(emulators)
        self.region = region
        self.functions_port = functions_port
        self.firestore_port = firestore_port
        self.auth_port = auth_port
        self.startup_timeout = startup_timeout
        self.shutdown_timeout = shutdown_timeout
        self.reuse_running = reuse_running

        self._import_dir = Path(import_dir) if import_dir is not None else None
        self._export_dir = Path(export_on_exit) if export_on_exit is not None else None
        self._requested_log_path = Path(log_path) if log_path is not None else None

        self._provided_api_key = api_key
        self._api_key: Optional[str] = None
        self._proc: Optional[subprocess.Popen[bytes]] = None
        self._log_file = None
        self._log_path: Optional[Path] = None
        self._owned = False
        self._entered = False
        self._stopped = False

        # Attaching to an already-running suite (e.g. the Docker image) needs
        # no checkout; firebase.json only matters when we spawn the emulators.
        if not reuse_running and not (self.repo_root / "firebase.json").is_file():
            raise LocalBackendError(
                f"{self.repo_root} does not look like a Deliberate Lab checkout "
                "(no firebase.json)."
            )
        unknown = set(self.emulators) - {
            "functions",
            "firestore",
            "auth",
            "database",
            "storage",
            "hosting",
            "pubsub",
        }
        if unknown:
            raise LocalBackendError(f"Unknown emulator(s): {sorted(unknown)}")

    # -- public surface ---------------------------------------------------

    @property
    def base_url(self) -> str:
        """Base URL to hand to ``dl.Client(base_url=...)``."""
        self._require_entered()
        return (
            f"http://127.0.0.1:{self.functions_port}"
            f"/{self.project_id}/{self.region}/api/v1"
        )

    @property
    def api_key(self) -> str:
        self._require_entered()
        assert self._api_key is not None
        return self._api_key

    @property
    def log_path(self) -> Optional[Path]:
        """Where emulator stdout/stderr is being written, if we started it."""
        return self._log_path

    def client(self) -> Any:
        """Build a `deliberate_lab.Client` pointed at this backend."""
        self._require_entered()
        try:
            import deliberate_lab as dl
        except ImportError as exc:  # pragma: no cover
            raise LocalBackendError(
                "deliberate_lab is not installed. Try: pip install "
                "'git+https://github.com/PAIR-code/deliberate-lab.git"
                "#subdirectory=scripts'"
            ) from exc
        return dl.Client(base_url=self.base_url, api_key=self.api_key)

    # -- context management -----------------------------------------------

    def __enter__(self) -> "LocalBackend":
        if self._entered:
            raise LocalBackendError(
                "LocalBackend is single-use; create a new instance per with-block."
            )
        self._entered = True
        try:
            busy = [p for p in self._required_ports().values() if _port_is_open(p)]
            if busy and not self.reuse_running:
                raise LocalBackendError(
                    f"Port(s) {busy} already in use. Another emulator suite (or "
                    "run_locally.sh) is probably running. Stop it, or pass "
                    "reuse_running=True to attach to it."
                )
            if busy:
                self._owned = False
            else:
                self._spawn()
                self._owned = True

            self._wait_for_ports()
            self._api_key = (
                self._provided_api_key
                or os.environ.get("DL_API_KEY")
                or self._mint_api_key()
            )
            self._wait_for_api()
        except BaseException:
            self._stop()
            raise
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._stop()
        return False

    # -- process lifecycle ------------------------------------------------

    def _required_ports(self) -> Mapping[str, int]:
        table = {
            "functions": self.functions_port,
            "firestore": self.firestore_port,
            "auth": self.auth_port,
        }
        return {name: table[name] for name in self.emulators if name in table}

    def _spawn(self) -> None:
        npx = shutil.which("npx")
        if npx is None:
            raise LocalBackendError(
                "`npx` not found on PATH. Install Node (see the repo's .nvmrc)."
            )

        cmd = [
            npx,
            "firebase",
            "emulators:start",
            "--project",
            self.project_id,
            "--only",
            ",".join(self.emulators),
        ]
        if self._import_dir is not None:
            import_path = (self.repo_root / self._import_dir).resolve()
            if not import_path.is_dir():
                raise LocalBackendError(f"import_dir does not exist: {import_path}")
            cmd += ["--import", str(import_path)]
        if self._export_dir is not None:
            cmd += ["--export-on-exit", str((self.repo_root / self._export_dir).resolve())]

        if self._requested_log_path is not None:
            self._log_path = self._requested_log_path
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = self._log_path.open("wb")
        else:
            handle = tempfile.NamedTemporaryFile(
                prefix="dl-emulators-", suffix=".log", delete=False
            )
            self._log_path = Path(handle.name)
            self._log_file = handle

        # Put the child in its own process group so we can signal the whole
        # tree later: `npx firebase` forks a Node process per emulator, and
        # Firestore/Auth are Java children of those. Killing only the npx PID
        # orphans them and leaves the ports bound.
        popen_kwargs: dict[str, Any] = {
            "cwd": str(self.repo_root),
            "stdin": subprocess.DEVNULL,
            "stdout": self._log_file,
            "stderr": subprocess.STDOUT,
        }
        if _IS_WINDOWS:
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True

        self._proc = subprocess.Popen(cmd, **popen_kwargs)
        # Backstop only: covers `python script.py` dying without unwinding.
        # The with-block is what normally performs cleanup.
        atexit.register(self._stop)

    def _stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        try:
            atexit.unregister(self._stop)
        except Exception:  # pragma: no cover - defensive
            pass

        proc, self._proc = self._proc, None
        if proc is not None and self._owned and proc.poll() is None:
            _terminate_tree(proc, self.shutdown_timeout)
        if proc is not None:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:  # pragma: no cover
                pass

        if self._log_file is not None:
            try:
                self._log_file.close()
            finally:
                self._log_file = None

    # -- readiness --------------------------------------------------------

    def _wait_for_ports(self) -> None:
        deadline = time.monotonic() + self.startup_timeout
        pending = dict(self._required_ports())
        while pending:
            self._assert_child_alive()
            for name, port in list(pending.items()):
                if _port_is_open(port):
                    del pending[name]
            if not pending:
                return
            if time.monotonic() > deadline:
                raise LocalBackendError(
                    f"Timed out after {self.startup_timeout:.0f}s waiting for "
                    f"emulator port(s) {pending}.{self._log_hint()}"
                )
            time.sleep(0.5)

    def _wait_for_api(self) -> None:
        """Poll GET /v1/health until it answers 200.

        The functions emulator binds its port before it has loaded the
        compiled functions, so an open socket is not readiness. A 200 here
        also proves the minted key resolves in Firestore, which makes this a
        genuine end-to-end probe.
        """
        deadline = time.monotonic() + self.startup_timeout
        last = "no response yet"
        while True:
            self._assert_child_alive()
            status, body = _http(
                "GET",
                f"{self.base_url}/health",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10.0,
            )
            if status == 200:
                return
            last = f"HTTP {status}: {body[:300]}" if status else body[:300]
            if status == 401:
                # Retry briefly (Firestore may still be importing), but this
                # usually means a stale key from $DL_API_KEY or a mismatched
                # project ID.
                pass
            if time.monotonic() > deadline:
                raise LocalBackendError(
                    f"Emulator is listening but {self.base_url}/health never "
                    f"returned 200. Last: {last}{self._log_hint()}"
                )
            time.sleep(1.0)

    def _assert_child_alive(self) -> None:
        """Fail fast instead of waiting out the timeout on a dead child."""
        if self._proc is not None and self._proc.poll() is not None:
            raise LocalBackendError(
                f"Emulator process exited with code {self._proc.returncode} "
                f"during startup.{self._log_hint()}"
            )

    def _log_hint(self) -> str:
        if self._log_path is None:
            return ""
        tail = _tail(self._log_path, 40)
        suffix = f"\nEmulator log: {self._log_path}"
        return f"{suffix}\n--- last lines ---\n{tail}" if tail else suffix

    # -- API key seeding --------------------------------------------------

    def _mint_api_key(self, name: str = "local-backend (python)") -> str:
        """Write a hashed API key straight into the Firestore emulator.

        The supported path is the web UI (Settings -> API Keys), which calls
        the `createDeliberateLabAPIKey` callable. That needs a signed-in
        experimenter, so for headless use we reproduce what
        `createDeliberateLabAPIKey` in dl_api_key.utils.ts writes:
        experimenters/{email}/apiKeys/{keyId} with a scrypt hash and salt.

        This couples us to backend internals that could change without notice.
        If the key ever stops being accepted, check that file first, or pass
        `api_key=` from a key you created in the UI. Never point this at a
        real project: it depends on the emulator's `Bearer owner` bypass of
        security rules.
        """
        if "firestore" not in self.emulators:
            raise LocalBackendError(
                "Cannot mint an API key without the firestore emulator; pass "
                "api_key= or include 'firestore' in emulators."
            )

        raw = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode()
        api_key = f"{_KEY_PREFIX}{raw}"
        key_id = hashlib.sha256(api_key.encode()).hexdigest()[:8]
        salt = secrets.token_hex(16)
        digest = hashlib.scrypt(
            api_key.encode(),
            salt=salt.encode(),  # hex digits as UTF-8, matching Node
            n=_SCRYPT_N,
            r=_SCRYPT_R,
            p=_SCRYPT_P,
            dklen=_SCRYPT_DKLEN,
        ).hex()

        document = {
            "fields": {
                "keyId": {"stringValue": key_id},
                "hash": {"stringValue": digest},
                "salt": {"stringValue": salt},
                "experimenterId": {"stringValue": self.experimenter_email},
                "name": {"stringValue": name},
                "permissions": {
                    "arrayValue": {
                        "values": [{"stringValue": "read"}, {"stringValue": "write"}]
                    }
                },
                "createdAt": {"integerValue": str(int(time.time() * 1000))},
                "lastUsed": {"nullValue": None},
            }
        }

        parent = urllib.parse.quote(self.experimenter_email, safe="")
        url = (
            f"http://127.0.0.1:{self.firestore_port}/v1/projects/{self.project_id}"
            f"/databases/(default)/documents/experimenters/{parent}/apiKeys"
            f"?documentId={key_id}"
        )
        status, body = _http(
            "POST",
            url,
            headers={
                "Authorization": "Bearer owner",  # emulator-only rules bypass
                "Content-Type": "application/json",
            },
            body=json.dumps(document).encode(),
            timeout=30.0,
        )
        if status != 200:
            raise LocalBackendError(
                f"Failed to seed API key into the Firestore emulator "
                f"(HTTP {status}): {body[:500]}"
            )
        return api_key

    def _require_entered(self) -> None:
        if not self._entered or self._stopped:
            raise LocalBackendError(
                "LocalBackend must be used as a context manager: "
                "`with LocalBackend(repo) as backend: ...`"
            )


# -- helpers --------------------------------------------------------------


def _port_is_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def _http(
    method: str,
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    body: bytes | None = None,
    timeout: float = 10.0,
) -> tuple[Optional[int], str]:
    """Returns (status, body). Status is None if the request never landed."""
    request = urllib.request.Request(
        url, data=body, method=method, headers=dict(headers or {})
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _terminate_tree(proc: subprocess.Popen[bytes], timeout: float) -> None:
    """Signal the whole process group, escalating if it does not go quietly."""
    if _IS_WINDOWS:  # pragma: no cover - platform specific
        try:
            proc.send_signal(signal.CTRL_BREAK_EVENT)
            proc.wait(timeout=timeout)
            return
        except (subprocess.TimeoutExpired, OSError, ValueError):
            pass
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            capture_output=True,
            check=False,
        )
        return

    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    # SIGTERM lets the Firebase CLI shut the emulators down cleanly (and honour
    # --export-on-exit) before we resort to SIGKILL.
    for sig, wait in ((signal.SIGTERM, timeout), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


def _tail(path: Path, lines: int) -> str:
    try:
        content = path.read_text("utf-8", "replace").splitlines()
    except OSError:
        return ""
    return "\n".join(content[-lines:])


if __name__ == "__main__":
    repo = sys.argv[1] if len(sys.argv) > 1 else "."
    with LocalBackend(repo, export_on_exit="emulator_test_config_local") as backend:
        print(f"backend up at {backend.base_url}")
        print(f"log: {backend.log_path}")
        client = backend.client()
        print("health:", client.health_check())
        print("experiments:", client.list_experiments().get("total"))
    print("backend stopped")
