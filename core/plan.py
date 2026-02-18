# core/plan.py
from dataclasses import dataclass
from typing import List


@dataclass
class ExecutionPlan:
    steps: List[str]
    step_idx: int = 0
    max_loops: int = 3

    def current_step(self) -> str:
        return self.steps[self.step_idx]

    def advance(self) -> None:
        self.step_idx += 1

    def is_finished(self) -> bool:
        return self.step_idx >= len(self.steps)

    def jump_to(self, step_name: str):
        if step_name not in self.steps:
            raise ValueError(f"{step_name} not in execution plan")
        self.step_idx = self.steps.index(step_name)


    def finish(self) -> None:
        """直接结束 plan"""
        self.step_idx = len(self.steps)
