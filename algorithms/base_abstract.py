from abc import ABC, abstractmethod

from dataset.models import Dataset, VmAssignment


class BaseAbstractScheduler(ABC):
    name: str

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def schedule(self, dataset: Dataset) -> list[VmAssignment]:
        raise NotImplementedError()

    @abstractmethod
    def run_time(self) -> float:
        raise NotImplementedError()

    @abstractmethod
    def decision_latency(self) -> float:
        raise NotImplementedError()
