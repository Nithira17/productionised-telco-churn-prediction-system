from typing import Optional, List
import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from spark_session import get_or_create_spark_session
logging.basicConfig(level=logging.INFO, format=
                    '%(asctime)s - %(levelname)s - %(message)s')

class OutlierDetectionStrategy(ABC):
    def __init__(self, spark: Optional[SparkSession] = None):
        self.spark = spark or get_or_create_spark_session()

    @abstractmethod
    def detect_outliers(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        pass

class IQROutlierDetector(OutlierDetectionStrategy):
    def __init__(self, threshold: float = 1.5, spark: Optional[SparkSession] = None):
        super().__init__(spark)
        self.threshold = threshold
        logging.info(f"Initialized IQROutlier Detection with threshold: {self.threshold}")

    def get_outlier_bounds(self, df, columns):
        bounds = {}
        for col in columns:
            quantiles = df.approxQuantile(col, [0.25, 0.75], 0.01)
            Q1, Q3 = quantiles[0], quantiles[1]
            IQR = Q3 - Q1

            lower_bound = Q1 - self.threshold * IQR
            upper_bound = Q3 + self.threshold * IQR

            bounds[col] = (lower_bound, upper_bound)

        return bounds
    
    def detect_outliers(self, df, columns):
        logging.info(f"\n{'='*60}")
        logging.info(f"OUTLIER DETECTION - IQR METHOD (PySpark)")
        logging.info(f"{'='*60}")
        logging.info(f"Starting IQR outlier detection for columns: {columns}")

        bounds = self.get_outlier_bounds(df, columns)

        result_df = df
        total_outliers = 0

        for col in columns:
            logging.info(f"\n--- Processing Column: {col} ---")

            lower_bound, upper_bound = bounds[col]
            outlier_col = f"{col}_outlier"

            result_df = result_df.withColumn(outlier_col,
                                             (F.col(col) < lower_bound) | (F.col(col) > upper_bound))
            
            outlier_count = result_df.filter(F.col(outlier_col)).count()
            total_rows = df.count()

        logging.info(f"\n{'='*60}")
        logging.info(f'✓ OUTLIER DETECTION COMPLETE - Total outlier instances: {total_outliers}')
        logging.info(f"{'='*60}\n")
        
        return result_df
    
# class STDOutlierDetector(OutlierDetectionStrategy):
#     def detect_outliers(self, df, columns):
#         outliers = pd.DataFrame(False, index=df.index, columns=columns)
#         for col in columns:
#             mean = df[col].mean()
#             std = df[col].std()

#             upper_limit = mean + 3 * std
#             lower_limit = mean - 3 * std

#             outliers[col] = (df[col] < lower_limit) | (df[col] > upper_limit)

#         logging.info('Outliers detected using STD method')
#         return outliers
    
class OutlierDetector:
    def __init__(self, strategy):
        self._strategy = strategy
        logging.info(f"Outlier Detector initialized with strategy: {strategy.__class__.__name__}")

    def detect_outliers(self, df, selected_columns):
        logging.info(f"Detecting outliers in {len(selected_columns)} columns")
        return self._strategy.detect_outliers(df, selected_columns)
    
    def handle_outliers(self, df, selected_columns, method='remove', min_outliers = 2):
        logging.info(f"\n{'='*60}")
        logging.info(f"OUTLIER HANDLING - {method.upper()} (PySpark)")
        logging.info(f"{'='*60}")
        logging.info(f"Handling outliers using method: {method}")

        initial_rows = df.count()

        df_with_outliers = self.detect_outliers(df, selected_columns)
        outlier_columns = [f"{col}_outlier" for col in selected_columns]
        outlier_count_expr = sum(F.col(col).cast('int') for col in outlier_columns)
        df_with_count = df_with_outliers.withColumn('outlier_count', outlier_count_expr)
        clean_df = df_with_count.filter(F.col("outlier_count") < min_outliers)
        clean_df = clean_df.drop("outlier_count")
        rows_removed = initial_rows - clean_df.count()

        logging.info(f"{'='*60}")
        return clean_df