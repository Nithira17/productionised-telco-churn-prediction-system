from typing import Optional
import pandas as pd
import logging
from abc import ABC, abstractmethod

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from spark_session import get_or_create_spark_session

logging.basicConfig(level=logging.INFO, format=
                    '%(asctime)s - %(levelname)s - %(message)s')

class FeatureBinningStrategy(ABC):
    def __init__(self, spark: Optional[SparkSession] = None):
        self.spark = spark or get_or_create_spark_session()

    @abstractmethod
    def bin_feature(self, df: pd.DataFrame, column: str) ->pd.DataFrame:
        pass

class CustomBinningStrategy(FeatureBinningStrategy):
    def __init__(self, bin_definitions, spark = None):
        super().__init__(spark)
        self.bin_definitions = bin_definitions
        logging.info(f"CustomBinningStratergy initialized with bins: {list(bin_definitions.keys())}")

    def bin_feature(self, df, column):
        bin_column = f'{column}Bins'
        case_expr = None

        for bin_label, bin_range in self.bin_definitions.items():
            if len(bin_range) == 2:
                condition = (F.col(column) >= bin_range[0]) & (F.col(column) <= bin_range[1])
            elif len(bin_range) == 1:
                condition = (F.col(column) >= bin_range[0])
            else:
                continue

            if case_expr is None:
                case_expr = F.when(condition, bin_label)
            else:
                case_expr = case_expr.when(condition, bin_label)
                
        df_binned = df.withColumn(bin_column, case_expr)
        df_binned = df_binned.drop(column)

        return df_binned