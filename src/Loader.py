from abc import ABC, abstractmethod
from pandas import DataFrame
class Loader(ABC):
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def get(self) -> DataFrame:
        pass



