import pandas as pd
import logging
from abc import ABC, abstractmethod
logging.basicConfig(level=logging.INFO, format=
                    '%(asctime)s - %(levelname)s - %(message)s')

class MissingValuesStrategy(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

class ReplaceValuesStrategy(MissingValuesStrategy):
    def __init__(self, replace_columns=[]):
        self.replace_columns = replace_columns
        logging.info(f'Replacing empty cells in {self.replace_columns} with 0 s')

    def handle(self, df):
        df[self.replace_columns[0]] = df[self.replace_columns[0]].replace(' ', '0')
        logging.info(f'empty cells replaced with 0s')
        return df