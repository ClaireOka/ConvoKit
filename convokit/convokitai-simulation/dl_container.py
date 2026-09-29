"""
dl_container.py — run the published Deliberate Lab backend image so that
`LocalBackend(..., reuse_running=True)` can attach to it.

Uses Docker when a daemon is available (your laptop) and falls back to
udocker, which runs images without a daemon or root privileges, where it is
not (Google Colab). udocker shares the host network, so the emulators are
reachable on localhost exactly as with `docker run -p`.

    from dl_container import BackendContainer
    from local_backend import LocalBackend

    with BackendContainer():
        with LocalBackend(".", reuse_running=True) as backend:
            client = backend.client()
            print(client.health_check())

In a notebook, `container = BackendContainer().start()` in one cell and
`container.stop()` in a later one works too.
"""

from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

__all__ = ["BackendContainer", "BackendContainerError", "DEFAULT_IMAGE", "FIRESTORE_PORT"]

# Pin a version tag rather than :latest so notebooks keep working when the
# backend image changes. v0.1 has Firestore on 8080; v0.2 onward uses 8085.
DEFAULT_IMAGE = "ghcr.io/claireoka/deliberate-lab-backend:v0.2"

# Must match emulators.firestore.port in the image's firebase.docker.json.
# Not 8080, because Colab's own runtime already listens there.
FIRESTORE_PORT = 8085

# The Firebase CLI gives the functions code 10s to load before it can list the
# functions to serve. Node is far slower under udocker (Colab), so without a
# longer limit the functions emulator comes up with no functions at all.
DEFAULT_ENV = {"FUNCTIONS_DISCOVERY_TIMEOUT": "300"}  # seconds

# udocker release downloaded when udocker isn't installed (e.g. pip can't install it)
UDOCKER_VERSION = "1.3.17"
_UDOCKER_URL = (
    "https://github.com/indigo-dc/udocker/releases/download/"
    f"{UDOCKER_VERSION}/udocker-{UDOCKER_VERSION}.tar.gz"
)
_UDOCKER_DIR = Path.home() / ".cache" / "dl-sim" / f"udocker-{UDOCKER_VERSION}"

_PORTS = (5001, FIRESTORE_PORT, 9099)  # functions, firestore, auth (firebase.docker.json)
_NAME = "dl-backend"


class BackendContainerError(RuntimeError):
    """Raised when the backend container cannot be started."""


class BackendContainer:
    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        *,
        runtime: Optional[str] = None,
        pull: bool = True,
        startup_timeout: float = 600.0,
        log_path: str | os.PathLike[str] | None = None,
        env: Optional[dict[str, str]] = None,
    ) -> None:
        """
        Args:
            image: Image reference to run.
            runtime: "docker" or "udocker". Defaults to docker when its daemon
                is reachable, otherwise udocker.
            pull: Pull the image before running. Set False to use an image
                that is already present locally.
            startup_timeout: Seconds to wait for the emulator ports. The first
                udocker run also extracts the image, which is slow.
            log_path: Where to write container output (udocker only; with
                Docker use `docker logs dl-backend`).
            env: Environment variables for the container, added to DEFAULT_ENV.
        """
        self.image = image
        self.runtime = runtime or ("docker" if _docker_available() else "udocker")
        if self.runtime not in ("docker", "udocker"):
            raise BackendContainerError(f"Unknown runtime: {self.runtime!r}")
        self.pull = pull
        self.env = {**DEFAULT_ENV, **(env or {})}
        self.startup_timeout = startup_timeout
        self.log_path = Path(log_path) if log_path else Path(
            tempfile.gettempdir()
        ) / "dl-backend.log"
        self._proc: Optional[subprocess.Popen[bytes]] = None
        self._started = False

    # -- public surface ---------------------------------------------------

    def start(self) -> "BackendContainer":
        busy = [p for p in _PORTS if _port_is_open(p)]
        if busy:
            raise BackendContainerError(
                f"Port(s) {busy} already in use; is a backend already running?"
            )
        if self.runtime == "docker":
            self._start_docker()
        else:
            self._start_udocker()
        self._started = True
        try:
            self._wait_for_ports()
        except BaseException:
            self.stop()
            raise
        return self

    def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        if self.runtime == "docker":
            subprocess.run(["docker", "stop", _NAME], capture_output=True, check=False)
            return
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            for sig, wait in ((signal.SIGTERM, 20.0), (signal.SIGKILL, 5.0)):
                try:
                    os.killpg(os.getpgid(proc.pid), sig)
                    proc.wait(timeout=wait)
                    break
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    continue

    def __enter__(self) -> "BackendContainer":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.stop()
        return False

    # -- runtimes ---------------------------------------------------------

    def _start_docker(self) -> None:
        if self.pull:
            _run(["docker", "pull", self.image])
        ports = [arg for p in _PORTS for arg in ("-p", f"{p}:{p}")]
        env = [arg for k, v in self.env.items() for arg in ("-e", f"{k}={v}")]
        _run(["docker", "run", "-d", "--rm", "--name", _NAME, *ports, *env, self.image])

    def _start_udocker(self) -> None:
        udocker = _udocker_command() or _download_udocker()
        # Colab runs everything as root, which udocker refuses by default.
        base = udocker + (["--allow-root"] if os.geteuid() == 0 else [])
        _run(base + ["install"])  # one-time download of udocker's engines
        if self.pull:
            _run(base + ["pull", self.image])
        # Reuse the extracted container across runs; extraction is the slow
        # part. Emulator state is in memory, so each run still starts fresh.
        if subprocess.run(base + ["inspect", _NAME], capture_output=True).returncode:
            _run(base + ["create", f"--name={_NAME}", self.image])

        log = self.log_path.open("wb")
        self._proc = subprocess.Popen(
            base + ["run", *(f"--env={k}={v}" for k, v in self.env.items()), _NAME],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log.close()

    # -- readiness --------------------------------------------------------

    def _wait_for_ports(self) -> None:
        # All ports must be open before LocalBackend(reuse_running=True) runs,
        # or it will conclude nothing is running and try to spawn emulators.
        # Docker Desktop (macOS) accepts connections on published ports as soon as the container
        # starts and resets them until the emulator inside listens, so an open port isn't enough.
        deadline = time.monotonic() + self.startup_timeout
        while not all(_port_is_serving(p) for p in _PORTS):
            if self._proc is not None and self._proc.poll() is not None:
                raise BackendContainerError(
                    f"Container exited with code {self._proc.returncode}.{self._log_hint()}"
                )
            if time.monotonic() > deadline:
                raise BackendContainerError(
                    f"Timed out after {self.startup_timeout:.0f}s waiting for "
                    f"ports {_PORTS}.{self._log_hint()}"
                )
            time.sleep(1.0)

    def _log_hint(self) -> str:
        if self.runtime == "docker":
            return f"\nSee: docker logs {_NAME}"
        try:
            tail = self.log_path.read_text("utf-8", "replace").splitlines()[-30:]
        except OSError:
            return ""
        return f"\nLog: {self.log_path}\n" + "\n".join(tail)


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


def _udocker_command() -> Optional[list[str]]:
    """The command that runs udocker, or None if it is not installed.

    `pip install udocker` only provides a `udocker` script, which isn't on PATH
    when pip's script directory isn't (e.g. a notebook kernel whose Python
    differs from the shell's), so also look next to this Python, and fall back
    to running udocker's entry point with this Python.
    """
    script = shutil.which("udocker") or shutil.which(
        "udocker", path=str(Path(sys.executable).parent)
    )
    if script is not None:
        return [script]
    try:
        import udocker.maincmd  # noqa: F401
    except ImportError:
        return None
    return [sys.executable, "-c", "import sys; from udocker.maincmd import main; sys.exit(main())"]


def _download_udocker() -> list[str]:
    """Download the udocker release (once) and return the command that runs it.

    This is udocker's documented install without pip: the release tarball
    contains the udocker package with a launcher script, run with this Python.
    """
    launcher = _UDOCKER_DIR / f"udocker-{UDOCKER_VERSION}" / "udocker" / "maincmd.py"
    if not launcher.exists():
        _UDOCKER_DIR.mkdir(parents=True, exist_ok=True)
        archive = _UDOCKER_DIR / "udocker.tar.gz"
        try:
            urllib.request.urlretrieve(_UDOCKER_URL, archive)
            with tarfile.open(archive) as tar:
                tar.extractall(_UDOCKER_DIR, filter="data")
        except (OSError, tarfile.TarError) as e:
            raise BackendContainerError(
                f"No Docker daemon and udocker is not installed, and downloading udocker from "
                f"{_UDOCKER_URL} failed: {e}\nInstall it with `%pip install udocker` and try again."
            ) from e
        finally:
            archive.unlink(missing_ok=True)
    return [sys.executable, str(launcher)]


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise BackendContainerError(
            f"{' '.join(cmd)} failed ({result.returncode}):\n{result.stdout}{result.stderr}"
        )


def _port_is_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def _port_is_serving(port: int, host: str = "127.0.0.1") -> bool:
    """Whether an HTTP server answers on the port (any status code counts)."""
    try:
        urllib.request.urlopen(f"http://{host}:{port}/", timeout=2).close()
    except urllib.error.HTTPError:
        return True
    except (OSError, ValueError):  # refused, reset, timed out, or not HTTP yet
        return False
    return True
