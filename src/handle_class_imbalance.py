import logging
import pandas as pd
from enum import Enum
from typing import Tuple
from abc import ABC, abstractmethod
from imblearn.over_sampling import SMOTE
logging.basicConfig(level=logging.INFO, format=
                    '%(asctime)s - %(levelname)s - %(message)s')


class HandlingImbalanceStrategy(ABC):
    @abstractmethod
    def handle(self, X: pd.DataFrame, Y: pd.Series) ->Tuple[pd.DataFrame, pd.Series]:
        pass

class HandleImbalanceMethod(str, Enum):
    SMOTE = 'smote'

class SMOTEHandleImbalanceStrategy(HandlingImbalanceStrategy):
    def __init__(self, random_state):
        self.random_state = random_state

    def handle(self, X, Y):
        smote = SMOTE(random_state=self.random_state)
        X_train_resampled, Y_train_resampled = smote.fit_resample(X, Y)

        return X_train_resampled, Y_train_resampled