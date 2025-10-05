import logging
from enum import Enum
from typing import Tuple, Optional
from abc import ABC, abstractmethod
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from spark_session import get_or_create_spark_session

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DataSplittingStrategy(ABC):
    def __init__(self, spark: Optional[SparkSession] = None):
        self.spark = spark or get_or_create_spark_session()

    @abstractmethod
    def split(self, df: DataFrame, target_column: str) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
        pass


class SplitType(str, Enum):
    SIMPLE = "simple"
    STRATIFIED = "stratified"


class SimpleTrainTestSplitStrategy(DataSplittingStrategy):
    def __init__(self, test_size: float = 0.2, spark: Optional[SparkSession] = None):
        super().__init__(spark)
        self.test_size = test_size
        logging.info(f"SimpleTrainTestSplitStrategy initialized with test_size={self.test_size}")

    def split(self, df: DataFrame, target_column: str) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
        logging.info("Starting simple train-test split (PySpark)")

        # Split full DataFrame first
        train_ratio = 1 - self.test_size
        train_df, test_df = df.randomSplit([train_ratio, self.test_size], seed=42)

        # Separate features (X) and target (Y)
        feature_cols = [c for c in df.columns if c != target_column]

        X_train = train_df.select(feature_cols)
        Y_train = train_df.select(target_column)

        X_test = test_df.select(feature_cols)
        Y_test = test_df.select(target_column)

        logging.info(
            f"Train-Test split complete:\n"
            f"  Train: {train_df.count()} rows (X={len(feature_cols)} cols)\n"
            f"  Test:  {test_df.count()} rows (X={len(feature_cols)} cols)"
        )

        return X_train, X_test, Y_train, Y_test
