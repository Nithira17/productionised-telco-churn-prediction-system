import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod
logging.basicConfig(level=logging.INFO, format=
                    '%(asctime)s - %(levelname)s - %(message)s')

class OutlierDetectionStrategy(ABC):
    @abstractmethod
    def detect_outliers(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        pass

class IQROutlierDetector(OutlierDetectionStrategy):
    def detect_outliers(self, df, columns):
        outliers = pd.DataFrame(False, index=df.index, columns=columns)
        for col in columns:
            Q1 = np.percentile(df[col], 25)
            Q3 = np.percentile(df[col], 75)
            IQR = Q3 - Q1

            lower_limit = Q1 - 1.5 * IQR
            upper_limit = Q3 + 1.5 * IQR

            outliers[col] = (df[col] < lower_limit) | (df[col] > upper_limit)

        logging.info('Outliers detected using IQR method')
        return outliers
    
class STDOutlierDetector(OutlierDetectionStrategy):
    def detect_outliers(self, df, columns):
        outliers = pd.DataFrame(False, index=df.index, columns=columns)
        for col in columns:
            mean = df[col].mean()
            std = df[col].std()

            upper_limit = mean + 3 * std
            lower_limit = mean - 3 * std

            outliers[col] = (df[col] < lower_limit) | (df[col] > upper_limit)

        logging.info('Outliers detected using STD method')
        return outliers
    
class OutlierDetector:
    def __init__(self, strategy):
        self._strategy = strategy

    def detect_outliers(self, df, selected_columns):
        return self._strategy.detect_outliers(df, selected_columns)
    
    def handle_outliers(self, df, selected_columns, method='remove'):
        outliers = self.detect_outliers(df, selected_columns)
        outlier_count = outliers.sum(axis=1)
        rows_to_remove = outlier_count >= 2
        return df[~rows_to_remove]