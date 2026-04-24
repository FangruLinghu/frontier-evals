"""Apptainer-based ComputerInterface and ComputerRuntime for running PaperBench on HPC clusters."""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import structlog.stdlib

import chz
from nanoeval.solvers.computer_tasks.code_execution_interface import (
    ComputerConfiguration,
    ComputerInterface,
    ComputerRuntime,
    ExecutionResult,
)

logger = structlog.stdlib.get_logger(component=__name__)

# Container paths that we bind-mount from host directories
_CONTAINER_PATHS = {
    "/submission": "submission",
    "/output": "output",
    "/tmp": "tmp",
}


class ApptainerComputerInterface(ComputerInterface):
    """
    ComputerInterface implementation using Apptainer (Singularity).

    Uses bind mounts to map host directories to container paths.
    Each shell command runs via `apptainer exec`.
    """

    def __init__(self, sif_path: str, workdir: Path, use_gpu: bool = False):
        self.sif_path = sif_path
        self.workdir = workdir
        self.use_gpu = use_gpu

        # Create host directories that map to container paths
        self.submission_dir = workdir / "submission"
        self.output_dir = workdir / "output"
        self.tmp_dir = workdir / "tmp"
        for d in [self.submission_dir, self.output_dir, self.tmp_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _host_path_for(self, container_path: str) -> Path:
        """Translate a container-absolute path to the corresponding host path."""
        for cpath, dirname in _CONTAINER_PATHS.items():
            if container_path == cpath or container_path.startswith(cpath + "/"):
                rel = container_path[len(cpath) :].lstrip("/")
                return self.workdir / dirname / rel
        # Fallback: treat as relative to tmp
        return self.workdir / "tmp" / container_path.lstrip("/")

    def _build_command(self, cmd: str) -> list[str]:
        """Build the apptainer exec command with bind mounts."""
        args = ["apptainer", "exec"]
        if self.use_gpu:
            args.append("--nv")
        args.extend([
            "--bind", f"{self.submission_dir}:/submission",
            "--bind", f"{self.output_dir}:/output",
            "--bind", f"{self.tmp_dir}:/tmp",
            "--writable-tmpfs",
            self.sif_path,
            "bash", "-c", cmd,
        ])
        return args

    async def send_shell_command(self, cmd: str, *, idempotent: bool = False) -> ExecutionResult:
        args = self._build_command(cmd)
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        return ExecutionResult(
            output=stdout or b"",
            exit_code=proc.returncode or 0,
        )

    async def upload(self, file: bytes, destination: str) -> None:
        host_path = self._host_path_for(destination)
        host_path.parent.mkdir(parents=True, exist_ok=True)
        host_path.write_bytes(file)

    async def download(self, file: str) -> bytes:
        host_path = self._host_path_for(file)
        return host_path.read_bytes()

    async def disable_internet(self) -> None:
        pass  # No-op on HPC

    async def fetch_container_names(self) -> list[str]:
        return []

    async def stop(self) -> None:
        pass  # Cleanup handled by the runtime context manager


@chz.chz
class ApptainerComputerRuntime(ComputerRuntime):
    """ComputerRuntime that uses Apptainer instead of Docker."""

    sif_path: str = chz.field(doc="Path to the .sif container image")
    use_gpu: bool = chz.field(default=False, doc="Pass --nv flag for GPU access")
    keep_workdir: bool = chz.field(default=False, doc="Keep workdir after completion for debugging")

    async def _do_runtime_setup(
        self, task: ComputerConfiguration, computer: ComputerInterface
    ) -> None:
        pass  # No special runtime setup needed

    @asynccontextmanager
    async def _start_computer(
        self, task: ComputerConfiguration
    ) -> AsyncGenerator[ComputerInterface, None]:
        workdir = Path(tempfile.mkdtemp(prefix="paperbench_apptainer_"))
        logger.info(f"Created Apptainer workdir: {workdir}")
        try:
            computer = ApptainerComputerInterface(
                sif_path=self.sif_path,
                workdir=workdir,
                use_gpu=self.use_gpu,
            )
            yield computer
        finally:
            if self.keep_workdir:
                logger.info(f"Keeping workdir for debugging: {workdir}")
            else:
                shutil.rmtree(workdir, ignore_errors=True)
                logger.info(f"Cleaned up Apptainer workdir: {workdir}")
