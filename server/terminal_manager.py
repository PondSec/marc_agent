from __future__ import annotations

import os
import pwd
import pty
import signal
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock, Thread
from time import monotonic
from uuid import uuid4

from agent.models import utc_now
from config.settings import AppConfig


class TerminalSessionNotFoundError(RuntimeError):
    pass


class TerminalAccessDeniedError(RuntimeError):
    pass


@dataclass(slots=True)
class TerminalSnapshot:
    id: str
    cwd: str
    shell: str
    created_at: str
    updated_at: str
    status: str
    cursor: int
    output: str
    exit_code: int | None = None
    reset: bool = False


@dataclass(slots=True)
class _TerminalSession:
    id: str
    owner_user_id: str
    owner_email: str
    owner_ip: str | None
    cwd: str
    shell: str
    process: subprocess.Popen[bytes]
    master_fd: int
    created_at: str
    updated_at: str
    last_used_monotonic: float
    buffer: str = ""
    buffer_base_cursor: int = 0
    exit_code: int | None = None
    closed: bool = False
    lock: Lock = field(default_factory=Lock)

    @property
    def status(self) -> str:
        return "exited" if self.exit_code is not None else "running"

    @property
    def cursor(self) -> int:
        return self.buffer_base_cursor + len(self.buffer)


class TerminalManager:
    def __init__(self, config: AppConfig):
        self.config = config
        self._lock = Lock()
        self._sessions: dict[str, _TerminalSession] = {}
        self._idle_timeout_seconds = 600
        self._max_buffer_chars = 200_000
        self._max_sessions_total = 3
        self._max_sessions_per_owner = 1

    def create_session(
        self,
        *,
        owner_user_id: str,
        owner_email: str,
        owner_ip: str | None = None,
        cwd: str | None = None,
    ) -> TerminalSnapshot:
        self._prune_expired_sessions()
        self._close_owner_sessions(owner_user_id)
        working_dir = self._resolve_cwd(cwd)
        shell = self._default_shell()
        shell_command = self._shell_command(shell)
        master_fd, slave_fd = pty.openpty()
        env = self._build_env(shell)
        process = subprocess.Popen(
            shell_command,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            cwd=str(working_dir),
            env=env,
            close_fds=True,
            start_new_session=True,
        )
        os.close(slave_fd)

        now = utc_now()
        session = _TerminalSession(
            id=uuid4().hex[:12],
            owner_user_id=owner_user_id,
            owner_email=owner_email,
            owner_ip=owner_ip,
            cwd=str(working_dir),
            shell=shell,
            process=process,
            master_fd=master_fd,
            created_at=now,
            updated_at=now,
            last_used_monotonic=monotonic(),
        )
        reader = Thread(target=self._reader_loop, args=(session.id,), daemon=True)
        with self._lock:
            self._evict_if_needed_locked()
            self._sessions[session.id] = session
        reader.start()
        return self.read(session.id, owner_user_id=owner_user_id, cursor=0)

    def read(self, session_id: str, *, owner_user_id: str, cursor: int = 0) -> TerminalSnapshot:
        self._prune_expired_sessions()
        session = self._require_session_for_owner(session_id, owner_user_id=owner_user_id)
        with session.lock:
            session.last_used_monotonic = monotonic()
            reset = cursor < session.buffer_base_cursor
            if reset:
                output = session.buffer
            else:
                output = session.buffer[cursor - session.buffer_base_cursor :]
            return TerminalSnapshot(
                id=session.id,
                cwd=session.cwd,
                shell=session.shell,
                created_at=session.created_at,
                updated_at=session.updated_at,
                status=session.status,
                cursor=session.cursor,
                output=output,
                exit_code=session.exit_code,
                reset=reset,
            )

    def write(self, session_id: str, owner_user_id: str, data: str) -> TerminalSnapshot:
        if not data:
            return self.read(session_id, owner_user_id=owner_user_id)
        session = self._require_session_for_owner(session_id, owner_user_id=owner_user_id)
        with session.lock:
            session.last_used_monotonic = monotonic()
            if session.exit_code is not None:
                return self._snapshot_locked(session, output="", reset=False)
            os.write(session.master_fd, data.encode("utf-8", errors="ignore"))
            session.updated_at = utc_now()
            return self._snapshot_locked(session, output="", reset=False)

    def interrupt(self, session_id: str, owner_user_id: str) -> TerminalSnapshot:
        session = self._require_session_for_owner(session_id, owner_user_id=owner_user_id)
        with session.lock:
            session.last_used_monotonic = monotonic()
            if session.exit_code is None:
                try:
                    session.process.send_signal(signal.SIGINT)
                except ProcessLookupError:
                    pass
                session.updated_at = utc_now()
            return self._snapshot_locked(session, output="", reset=False)

    def close(self, session_id: str, *, owner_user_id: str | None = None) -> None:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            return
        if owner_user_id and session.owner_user_id != owner_user_id:
            with self._lock:
                self._sessions[session_id] = session
            raise TerminalAccessDeniedError("Terminal session belongs to a different user.")
        self._close_session(session)

    def close_all(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            self._close_session(session)

    def _snapshot_locked(self, session: _TerminalSession, *, output: str, reset: bool) -> TerminalSnapshot:
        return TerminalSnapshot(
            id=session.id,
            cwd=session.cwd,
            shell=session.shell,
            created_at=session.created_at,
            updated_at=session.updated_at,
            status=session.status,
            cursor=session.cursor,
            output=output,
            exit_code=session.exit_code,
            reset=reset,
        )

    def _reader_loop(self, session_id: str) -> None:
        session = self._require_session(session_id)
        try:
            while True:
                try:
                    chunk = os.read(session.master_fd, 4096)
                except OSError:
                    chunk = b""
                if not chunk:
                    break
                text = chunk.decode("utf-8", errors="replace")
                with session.lock:
                    session.buffer += text
                    session.updated_at = utc_now()
                    session.last_used_monotonic = monotonic()
                    if len(session.buffer) > self._max_buffer_chars:
                        overflow = len(session.buffer) - self._max_buffer_chars
                        session.buffer = session.buffer[overflow:]
                        session.buffer_base_cursor += overflow
            exit_code = session.process.wait(timeout=5)
        except Exception:
            exit_code = session.process.poll()
            if exit_code is None:
                exit_code = -1
        finally:
            with session.lock:
                session.exit_code = exit_code
                session.updated_at = utc_now()
            try:
                os.close(session.master_fd)
            except OSError:
                pass

    def _close_session(self, session: _TerminalSession) -> None:
        with session.lock:
            if session.closed:
                return
            session.closed = True
            if session.process.poll() is None:
                try:
                    session.process.terminate()
                    session.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    session.process.kill()
                    session.process.wait(timeout=2)
                except ProcessLookupError:
                    pass
            session.exit_code = session.process.poll()
        try:
            os.close(session.master_fd)
        except OSError:
            pass

    def _require_session(self, session_id: str) -> _TerminalSession:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise TerminalSessionNotFoundError("Terminal session not found.")
        return session

    def _require_session_for_owner(self, session_id: str, *, owner_user_id: str) -> _TerminalSession:
        session = self._require_session(session_id)
        if session.owner_user_id != owner_user_id:
            raise TerminalAccessDeniedError("Terminal session belongs to a different user.")
        return session

    def _close_owner_sessions(self, owner_user_id: str) -> None:
        stale: list[_TerminalSession] = []
        with self._lock:
            for session_id, session in list(self._sessions.items()):
                if session.owner_user_id == owner_user_id:
                    stale.append(self._sessions.pop(session_id))
        for session in stale:
            self._close_session(session)

    def _evict_if_needed_locked(self) -> None:
        if len(self._sessions) < self._max_sessions_total:
            return
        oldest_session = min(self._sessions.values(), key=lambda item: item.last_used_monotonic)
        self._sessions.pop(oldest_session.id, None)
        self._close_session(oldest_session)

    def _prune_expired_sessions(self) -> None:
        now = monotonic()
        expired: list[_TerminalSession] = []
        with self._lock:
            for session_id, session in list(self._sessions.items()):
                if now - session.last_used_monotonic > self._idle_timeout_seconds:
                    expired.append(self._sessions.pop(session_id))
        for session in expired:
            self._close_session(session)

    def _resolve_cwd(self, cwd: str | None) -> Path:
        if cwd:
            target = Path(cwd).expanduser().resolve()
            if target.exists() and target.is_dir():
                return target
        return self.config.workspace_path

    def _default_shell(self) -> str:
        for candidate in (os.environ.get("SHELL"), "/bin/bash", "/bin/zsh", "/bin/sh"):
            if candidate and Path(candidate).exists():
                return candidate
        return "/bin/sh"

    def _shell_command(self, shell: str) -> list[str]:
        name = Path(shell).name
        if name == "bash":
            return [shell, "--noprofile", "--norc", "-i"]
        if name == "zsh":
            return [shell, "-f", "-i"]
        return [shell, "-i"]

    def _build_env(self, shell: str) -> dict[str, str]:
        user_name = pwd.getpwuid(os.getuid()).pw_name
        home = str(Path.home())
        return {
            "TERM": "xterm-256color",
            "COLORTERM": "truecolor",
            "HOME": home,
            "USER": user_name,
            "LOGNAME": user_name,
            "SHELL": shell,
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PWD": home,
        }
