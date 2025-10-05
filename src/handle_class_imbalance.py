import logging
from enum import Enum
from typing import Tuple, Optional
from abc import ABC, abstractmethod

from pyspark.sql import DataFrame, SparkSession
from imblearn.over_sampling import SMOTE
from spark_session import get_or_create_spark_session
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class HandlingImbalanceStrategy(ABC):
    def __init__(self, spark: Optional[SparkSession] = None):
        self.spark = spark or get_or_create_spark_session()

    @abstractmethod
    def handle(self, X: DataFrame, y_col: str) -> Tuple[DataFrame, DataFrame]:
        pass


class HandleImbalanceMethod(str, Enum):
    SMOTE = "smote"


class SMOTEHandleImbalanceStrategy(HandlingImbalanceStrategy):
    def __init__(self, random_state: int = 42, spark = None):
        super().__init__(spark)
        self.random_state = random_state
        logging.info(f"SMOTEHandleImbalanceStrategy initialized with random_state={self.random_state}")

    def handle(self, X: DataFrame, y_col: str) -> Tuple[DataFrame, DataFrame]:
        logging.info("Starting SMOTE handling (PySpark hybrid mode)")

        # Disable Arrow for more stable conversion with large datasets
        self.spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")
        
        try:
            # Convert Spark DataFrame to Pandas
            pdf = X.select("*").toPandas()
            logging.info(f"Converted to pandas DataFrame with shape {pdf.shape}")

            # Separate features and labels
            y = pdf[y_col]
            X_features = pdf.drop(columns=[y_col])

            # Apply SMOTE
            smote = SMOTE(random_state=self.random_state)
            X_res, y_res = smote.fit_resample(X_features, y)

            logging.info(f"SMOTE applied – original shape {X_features.shape}, new shape {X_res.shape}")

            # Convert back to Spark DataFrames with chunking for large datasets
            X_resampled = self.spark.createDataFrame(pd.DataFrame(X_res, columns=X_features.columns))
            y_resampled = self.spark.createDataFrame(pd.DataFrame(y_res, columns=[y_col]))

            logging.info("Converted resampled data back to PySpark DataFrames")

            return X_resampled, y_resampled
        
        finally:
            # Re-enable Arrow
            self.spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
