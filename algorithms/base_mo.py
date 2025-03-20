from abc import abstractmethod
from algorithms.base_static import BaseStaticScheduler
from dataset.models import Dataset

SolutionStoreType = dict[int, list[list[tuple[int, int]]]]


class BaseMoScheduler(BaseStaticScheduler):
    def __init__(self, name: str, store: SolutionStoreType, index: int):
        super().__init__(name)
        self.store = store
        self.index = index

    def compute_assignments(self, dataset: Dataset) -> list[tuple[int, int]]:
        if dataset.key not in self.store:
            self.store[dataset.key] = self.get_pareto_solutions(dataset)

        pareto_solutions = self.store[dataset.key]
        return pareto_solutions[self.index % len(pareto_solutions)]

    @abstractmethod
    def get_pareto_solutions(self, dataset: Dataset) -> list[list[tuple[int, int]]]:
        raise NotImplementedError()
