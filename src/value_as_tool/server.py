"""Command construction and lifecycle helpers for local model servers.

Importing this module and building commands are side-effect free. A process is
started only by an explicit call to :meth:`ModelServer.start`.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO, Any, Literal
from urllib.parse import urlparse

from value_as_tool.config import ExperimentConfig

Backend = Literal["sglang", "vllm"]
Role = Literal["solver", "judge"]
SUPPORTED_SGLANG_VERSION = "0.5.19"
SUPPORTED_VLLM_VERSION = "0.29.0"


def _executable(env_name: str, default: str) -> str:
    return os.environ.get(env_name) or shutil.which(default) or default


def _reject_speculative_decoding(extra_args: Sequence[str]) -> None:
    for argument in extra_args:
        option = argument.lstrip("-").replace("_", "-")
        if option.startswith(("speculative", "spec-decode")):
            raise ValueError(
                "speculative decoding is incompatible with exact per-call "
                "thinking-budget enforcement"
            )


def build_sglang_command(
    model_path: str | Path,
    served_model_name: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    context_length: int = 262_144,
    tensor_parallel_size: int = 1,
    mem_fraction_static: float = 0.8,
    executable: str | None = None,
    extra_args: Sequence[str] = (),
) -> list[str]:
    """Build the supported Qwen-family SGLang invocation."""

    if not 0 < port < 65_536 or context_length <= 0 or tensor_parallel_size <= 0:
        raise ValueError("invalid port, context length, or tensor parallel size")
    if not 0 < mem_fraction_static <= 1:
        raise ValueError("mem_fraction_static must be in (0, 1]")
    _reject_speculative_decoding(extra_args)
    return [
        executable or _executable("VALUE_AS_TOOL_SGLANG_BIN", "sglang"),
        "serve",
        "--model-path",
        str(model_path),
        "--served-model-name",
        served_model_name,
        "--host",
        host,
        "--port",
        str(port),
        "--context-length",
        str(context_length),
        "--tp-size",
        str(tensor_parallel_size),
        "--reasoning-parser",
        "qwen3",
        "--tool-call-parser",
        "qwen3_coder",
        "--mem-fraction-static",
        str(mem_fraction_static),
        # Required for the per-call thinking budget: without it SGLang drops the
        # request's custom_logit_processor and thinking runs to the token cap.
        "--enable-custom-logit-processor",
        # Qwen hybrid GDN attention is more reliable with this combination
        # than FlashInfer on current Blackwell deployments.  Do not enable
        # NEXTN/EAGLE speculative decoding here: SGLang's speculative path does
        # not reliably apply a per-request custom logit processor to every
        # drafted token.  That silently defeats the thinking budget above and
        # lets reasoning consume the complete response allowance.
        "--attention-backend",
        "triton",
        "--mamba-radix-cache-strategy",
        "extra_buffer",
        *extra_args,
    ]


def build_vllm_command(
    model_path: str | Path,
    served_model_name: str,
    *,
    role: Role = "solver",
    host: str = "127.0.0.1",
    port: int = 8000,
    context_length: int = 262_144,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_num_seqs: int = 16,
    executable: str | None = None,
    extra_args: Sequence[str] = (),
) -> list[str]:
    """Build a vLLM fallback for Qwen or the GPT-OSS external judge."""

    if role not in ("solver", "judge"):
        raise ValueError(f"unsupported server role: {role}")
    if not 0 < port < 65_536 or context_length <= 0 or tensor_parallel_size <= 0:
        raise ValueError("invalid port, context length, or tensor parallel size")
    if not 0 < gpu_memory_utilization <= 1 or max_num_seqs <= 0:
        raise ValueError("invalid GPU utilization or sequence count")
    _reject_speculative_decoding(extra_args)
    command = [
        executable or _executable("VALUE_AS_TOOL_VLLM_BIN", "vllm"),
        "serve",
        str(model_path),
        "--served-model-name",
        served_model_name,
        "--host",
        host,
        "--port",
        str(port),
        "--max-model-len",
        str(context_length),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-num-seqs",
        str(max_num_seqs),
        "--enable-prefix-caching",
        "--trust-remote-code",
        "--reasoning-parser",
        "qwen3" if role == "solver" else "openai_gptoss",
    ]
    if role == "solver":
        command.extend(
            [
                "--enable-auto-tool-choice",
                "--tool-call-parser",
                "qwen3_coder",
                "--language-model-only",
            ]
        )
    else:
        # Match the known-working GPT-OSS direct-evaluation profile. The
        # auto-selected FlashInfer MXFP4 path may attempt a first-run kernel
        # download on compute nodes without egress.
        command.extend(["--moe-backend", "triton"])
    command.extend(extra_args)
    return command


def build_server_command(
    backend: Backend,
    model_path: str | Path,
    served_model_name: str,
    *,
    role: Role = "solver",
    **kwargs: Any,
) -> list[str]:
    if backend == "sglang":
        if role != "solver":
            raise ValueError("the supported SGLang profile is solver-only")
        return build_sglang_command(model_path, served_model_name, **kwargs)
    if backend == "vllm":
        return build_vllm_command(model_path, served_model_name, role=role, **kwargs)
    raise ValueError(f"unsupported backend: {backend!r}")


@dataclass(frozen=True, slots=True)
class ServerSpec:
    backend: Backend
    role: Role
    model_path: str
    served_model_name: str
    base_url: str
    log_path: Path
    context_length: int
    tensor_parallel_size: int = 1
    startup_timeout_seconds: float = 900
    extra_args: tuple[str, ...] = ()
    environment: Mapping[str, str] = field(default_factory=dict)

    @property
    def host(self) -> str:
        return urlparse(self.base_url).hostname or "127.0.0.1"

    @property
    def port(self) -> int:
        parsed = urlparse(self.base_url)
        if parsed.port is not None:
            return parsed.port
        return 443 if parsed.scheme == "https" else 80

    def command(self) -> list[str]:
        return build_server_command(
            self.backend,
            self.model_path,
            self.served_model_name,
            role=self.role,
            host=self.host,
            port=self.port,
            context_length=self.context_length,
            tensor_parallel_size=self.tensor_parallel_size,
            extra_args=self.extra_args,
        )


def endpoint_models(
    base_url: str,
    *,
    api_key: str = "EMPTY",
    timeout_seconds: float = 5,
) -> set[str]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        value = json.load(response)
    if not isinstance(value, Mapping) or not isinstance(value.get("data"), list):
        raise RuntimeError("model endpoint returned an invalid /models payload")
    return {
        str(item["id"])
        for item in value["data"]
        if isinstance(item, Mapping) and item.get("id") is not None
    }


def endpoint_ready(
    base_url: str,
    served_model_name: str,
    *,
    api_key: str = "EMPTY",
    timeout_seconds: float = 5,
) -> bool:
    try:
        return served_model_name in endpoint_models(
            base_url, api_key=api_key, timeout_seconds=timeout_seconds
        )
    except (OSError, RuntimeError, ValueError, urllib.error.URLError, json.JSONDecodeError):
        return False


class ModelServer(AbstractContextManager["ModelServer"]):
    """Own one explicitly started subprocess and terminate only its process group."""

    def __init__(self, spec: ServerSpec) -> None:
        self.spec = spec
        self.process: subprocess.Popen[bytes] | None = None
        self._log_handle: IO[bytes] | None = None

    @property
    def command(self) -> list[str]:
        return self.spec.command()

    def start(self) -> ModelServer:
        if self.process is not None:
            raise RuntimeError("server has already been started")
        if self.spec.host in ("127.0.0.1", "localhost", "0.0.0.0", "::1"):
            probe_host = "127.0.0.1" if self.spec.host == "0.0.0.0" else self.spec.host
            try:
                with socket.create_connection((probe_host, self.spec.port), timeout=0.2):
                    raise RuntimeError(f"port {self.spec.port} is already in use")
            except (ConnectionRefusedError, TimeoutError, OSError):
                pass
        self.spec.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.spec.log_path.open("ab")
        environment = os.environ.copy()
        environment.update(self.spec.environment)
        try:
            self.process = subprocess.Popen(
                self.command,
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=environment,
            )
        except BaseException:
            self._log_handle.close()
            self._log_handle = None
            raise
        try:
            self.wait_ready()
        except BaseException:
            self.stop()
            raise
        return self

    def wait_ready(self) -> None:
        if self.process is None:
            raise RuntimeError("server is not started")
        deadline = time.monotonic() + self.spec.startup_timeout_seconds
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"model server exited with {self.process.returncode}; see {self.spec.log_path}"
                )
            if endpoint_ready(self.spec.base_url, self.spec.served_model_name):
                return
            time.sleep(2)
        raise TimeoutError(
            f"server did not become ready in {self.spec.startup_timeout_seconds}s; "
            f"see {self.spec.log_path}"
        )

    def stop(self, *, grace_seconds: float = 30) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=grace_seconds)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=10)
        if self._log_handle is not None:
            self._log_handle.close()
        self.process = None
        self._log_handle = None

    def __enter__(self) -> ModelServer:
        return self.start()

    def __exit__(self, exc_type, exc, traceback) -> None:  # type: ignore[no-untyped-def]
        self.stop()


class SGLangServer(ModelServer):
    def __init__(
        self,
        *,
        model_path: str | Path,
        served_model_name: str,
        base_url: str = "http://127.0.0.1:8000/v1",
        log_path: str | Path = "artifacts/server-solver.log",
        context_length: int = 262_144,
        tensor_parallel_size: int = 1,
        startup_timeout_seconds: float = 900,
        extra_args: Sequence[str] = (),
    ) -> None:
        super().__init__(
            ServerSpec(
                backend="sglang",
                role="solver",
                model_path=str(model_path),
                served_model_name=served_model_name,
                base_url=base_url,
                log_path=Path(log_path),
                context_length=context_length,
                tensor_parallel_size=tensor_parallel_size,
                startup_timeout_seconds=startup_timeout_seconds,
                extra_args=tuple(extra_args),
            )
        )


class VLLMServer(ModelServer):
    def __init__(
        self,
        *,
        model_path: str | Path,
        served_model_name: str,
        role: Role = "solver",
        base_url: str = "http://127.0.0.1:8000/v1",
        log_path: str | Path = "artifacts/server.log",
        context_length: int = 262_144,
        tensor_parallel_size: int = 1,
        startup_timeout_seconds: float = 900,
        extra_args: Sequence[str] = (),
    ) -> None:
        super().__init__(
            ServerSpec(
                backend="vllm",
                role=role,
                model_path=str(model_path),
                served_model_name=served_model_name,
                base_url=base_url,
                log_path=Path(log_path),
                context_length=context_length,
                tensor_parallel_size=tensor_parallel_size,
                startup_timeout_seconds=startup_timeout_seconds,
                extra_args=tuple(extra_args),
                environment={"VLLM_USE_RUST_FRONTEND": "1"} if role == "solver" else {},
            )
        )


@dataclass(frozen=True, slots=True)
class PreflightCheck:
    name: str
    ok: bool
    detail: str
    required: bool = True


@dataclass(frozen=True, slots=True)
class PreflightReport:
    checks: tuple[PreflightCheck, ...]

    @property
    def ok(self) -> bool:
        return all(check.ok or not check.required for check in self.checks)

    @property
    def errors(self) -> tuple[str, ...]:
        return tuple(check.detail for check in self.checks if check.required and not check.ok)

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(check.detail for check in self.checks if not check.required and not check.ok)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "checks": [asdict(check) for check in self.checks]}

    def require_ok(self) -> None:
        if not self.ok:
            raise RuntimeError("preflight failed: " + "; ".join(self.errors))


def _binary_available(command: str) -> bool:
    path = Path(command)
    if path.is_absolute() or "/" in command:
        return path.is_file() and os.access(path, os.X_OK)
    return shutil.which(command) is not None


def preflight(
    config: ExperimentConfig,
    *,
    experiment_fingerprint: str | None = None,
    solver_model_path: str | Path | None = None,
    judge_model_path: str | Path | None = None,
    check_endpoints: bool = False,
    require_server_binaries: bool = False,
    environment: Mapping[str, str] | None = None,
) -> PreflightReport:
    """Inspect local prerequisites without downloading or starting anything."""

    environ = os.environ if environment is None else environment
    checks: list[PreflightCheck] = []
    selected_fingerprint = experiment_fingerprint or config.fingerprint
    checks.append(
        PreflightCheck(
            "experiment_fingerprint",
            len(selected_fingerprint) == 64,
            f"experiment fingerprint: {selected_fingerprint}",
        )
    )
    backend_env = (
        "VALUE_AS_TOOL_SGLANG_BIN"
        if config.runtime.solver_backend == "sglang"
        else "VALUE_AS_TOOL_VLLM_BIN"
    )
    backend_default = "sglang" if config.runtime.solver_backend == "sglang" else "vllm"
    solver_binary = environ.get(backend_env) or shutil.which(backend_default) or backend_default
    checks.append(
        PreflightCheck(
            "solver_binary",
            _binary_available(solver_binary),
            f"solver backend executable: {solver_binary}",
            required=require_server_binaries,
        )
    )
    judge_binary = environ.get("VALUE_AS_TOOL_VLLM_BIN") or shutil.which("vllm") or "vllm"
    checks.append(
        PreflightCheck(
            "judge_binary",
            _binary_available(judge_binary),
            f"judge backend executable: {judge_binary}",
            required=require_server_binaries,
        )
    )
    for name, path in (("solver_model", solver_model_path), ("judge_model", judge_model_path)):
        if path is not None:
            source = Path(path)
            checks.append(
                PreflightCheck(name, source.exists(), f"{name} path: {source}", required=True)
            )
    for name, model in (("solver", config.models.solver), ("judge", config.models.judge)):
        present = bool(environ.get(model.api_key_env))
        base_url = config.models.operational_base_url(name, environ)
        local = (urlparse(base_url).hostname or "") in {
            "127.0.0.1",
            "localhost",
            "::1",
        }
        checks.append(
            PreflightCheck(
                f"{name}_api_key",
                present or local,
                f"{model.api_key_env} {'is set' if present else 'is not set'}",
                required=not local,
            )
        )
        if check_endpoints:
            api_key = environ.get(model.api_key_env, "EMPTY")
            ready = endpoint_ready(base_url, model.name, api_key=api_key)
            checks.append(
                PreflightCheck(
                    f"{name}_endpoint",
                    ready,
                    f"{base_url} {'serves' if ready else 'does not serve'} {model.name}",
                )
            )
    return PreflightReport(tuple(checks))


def exec_server(spec: ServerSpec) -> None:
    """Replace the current process with the configured server (CLI helper)."""

    command = spec.command()
    environment = os.environ.copy()
    environment.update(spec.environment)
    os.execvpe(command[0], command, environment)


__all__ = [
    "ModelServer",
    "PreflightCheck",
    "PreflightReport",
    "SGLangServer",
    "ServerSpec",
    "SUPPORTED_SGLANG_VERSION",
    "SUPPORTED_VLLM_VERSION",
    "VLLMServer",
    "build_server_command",
    "build_sglang_command",
    "build_vllm_command",
    "endpoint_models",
    "endpoint_ready",
    "exec_server",
    "preflight",
]
