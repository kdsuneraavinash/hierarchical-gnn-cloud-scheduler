from abc import abstractmethod
from dataclasses import dataclass
from algorithms.base_static import BaseStaticScheduler
from dataset.models import Dataset


@dataclass
class SolutionStore:
    solutions: list[list[tuple[int, int]]] | None = None


class BaseMoScheduler(BaseStaticScheduler):
    def __init__(self, name: str, store: SolutionStore, index: int):
        super().__init__(name)
        self.store = store
        self.index = index

    def compute_assignments(self, dataset: Dataset) -> list[tuple[int, int]]:
        if self.store.solutions is None:
            self.store.solutions = self.get_pareto_solutions(dataset)

        return self.store.solutions[self.index % len(self.store.solutions)]

    @abstractmethod
    def get_pareto_solutions(self, dataset: Dataset) -> list[list[tuple[int, int]]]:
        raise NotImplementedError()
