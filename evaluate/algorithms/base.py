from dataset.models import Dataset, VmAssignment


class BaseScheduler:
    name: str

    def __init__(self, name: str):
        self.name = name

    def schedule(self, dataset: Dataset) -> list[VmAssignment]:
        raise NotImplementedError()
