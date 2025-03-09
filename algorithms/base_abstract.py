from abc import ABC

from dataset.models import Dataset, VmAssignment


class BaseAbstractScheduler(ABC):
    name: str

    def __init__(self, name: str):
        self.name = name

    def schedule(self, dataset: Dataset) -> list[VmAssignment]:
        raise NotImplementedError()
