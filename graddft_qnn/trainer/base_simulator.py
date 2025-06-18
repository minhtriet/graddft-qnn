from abc import ABC, abstractmethod


class Simulator(ABC):
    @abstractmethod
    def simulate(self) -> tuple[float, list[float], list[float]]:
        pass

    def evaluate(self):
        pass
