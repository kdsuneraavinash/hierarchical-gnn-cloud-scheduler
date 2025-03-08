from dataset.models import VmAssignment


class BaseScheduler:
    name: str

    def __init__(self, name: str):
        self.name = name

    def schedule(self, dataset) -> list[VmAssignment]:
        raise NotImplementedError()
