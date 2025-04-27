from abc import ABC, abstractmethod
from pandas import DataFrame
class Loader(ABC):
    """
    Base class for differents type of Datasets
    """
    def __init__(self) -> None:
        """Initialize base loader"""
        super().__init__()
    
    @abstractmethod
    def get(self) -> DataFrame:
        """Load and get dataset from designed source"""
        pass



