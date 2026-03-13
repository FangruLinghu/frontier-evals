from simformer.tasks.base import Task, SimulatorTask, BenchmarkTask
from simformer.tasks.gaussian_linear import GaussianLinearTask
from simformer.tasks.gaussian_mixture import GaussianMixtureTask
from simformer.tasks.two_moons import TwoMoonsTask
from simformer.tasks.slcp import SLCPTask
from simformer.tasks.tree import TreeTask
from simformer.tasks.hmm import HMMTask
from simformer.tasks.lotka_volterra import LotkaVolterraTask
from simformer.tasks.sird import SIRDTask
from simformer.tasks.hodgkin_huxley import HodgkinHuxleyTask

__all__ = [
    "Task",
    "SimulatorTask",
    "BenchmarkTask",
    "GaussianLinearTask",
    "GaussianMixtureTask",
    "TwoMoonsTask",
    "SLCPTask",
    "TreeTask",
    "HMMTask",
    "LotkaVolterraTask",
    "SIRDTask",
    "HodgkinHuxleyTask",
]


def get_task(task_name: str, **kwargs):
    """
    Factory function to get a task by name.

    Args:
        task_name: Name of the task
        **kwargs: Additional arguments for the task

    Returns:
        Task instance
    """
    tasks = {
        "gaussian_linear": GaussianLinearTask,
        "gaussian_mixture": GaussianMixtureTask,
        "two_moons": TwoMoonsTask,
        "slcp": SLCPTask,
        "tree": TreeTask,
        "hmm": HMMTask,
        "lotka_volterra": LotkaVolterraTask,
        "sird": SIRDTask,
        "hodgkin_huxley": HodgkinHuxleyTask,
    }

    task_name = task_name.lower().replace("-", "_")

    if task_name not in tasks:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(tasks.keys())}")

    return tasks[task_name](**kwargs)
