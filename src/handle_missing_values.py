# from typing import Optional
# import pandas as pd
# import logging
# from abc import ABC, abstractmethod

# from pyspark.sql import SparkSession, DataFrame
# from pyspark.sql import functions as F
# from spark_session import get_or_create_spark_session

# logging.basicConfig(level=logging.INFO, format=
#                     '%(asctime)s - %(levelname)s - %(message)s')

# class MissingValuesStrategy(ABC):
#     def __init__(self, spark: Optional[SparkSession] = None):
#         self.spark = spark or get_or_create_spark_session()

#     @abstractmethod
#     def handle(self, df: pd.DataFrame) -> pd.DataFrame:
#         pass

# class ReplaceValuesStrategy(MissingValuesStrategy):
#     def __init__(self, replace_columns=[], spark: Optional[SparkSession] = None):
#         super().__init__(spark)
#         self.replace_columns = replace_columns
#         logging.info(f'Replacing empty cells in {self.replace_columns} with 0 s')

#     def handle(self, df):
#         df_cleaned = df.withColumn(
#                                     self.replace_columns[0],
#                                     F.when(F.col(self.replace_columns[0]) == ' ', F.lit('0'))
#                                     .otherwise(F.col(self.replace_columns[0]))
#                                     )

#         logging.info(f'empty cells in {self.replace_columns[0]} column replaced with 0s')
#         return df_cleaned

from typing import Optional
import logging
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from abc import ABC, abstractmethod
from spark_session import get_or_create_spark_session

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MissingValuesStrategy(ABC):
    def __init__(self, spark: Optional[SparkSession] = None):
        self.spark = spark or get_or_create_spark_session()

    @abstractmethod
    def handle(self, df: DataFrame) -> DataFrame:
        pass


class ReplaceValuesStrategy(MissingValuesStrategy):
    def __init__(self, replace_columns=[], spark: Optional[SparkSession] = None):
        super().__init__(spark)
        self.replace_columns = replace_columns
        logger.info(f"Replacing empty/invalid cells in {self.replace_columns} with 0s")

    def handle(self, df):
        df_cleaned = df
        for col in self.replace_columns:
            df_cleaned = (
                df_cleaned
                # Trim spaces and normalize blanks
                .withColumn(col, F.trim(F.col(col)))
                # Replace nulls, blanks, or non-numeric strings
                .withColumn(
                    col,
                    F.when(
                        (F.col(col).isNull()) | (F.col(col) == "") | (F.col(col).rlike("^[^0-9.]+$")),
                        F.lit("0")
                    ).otherwise(F.col(col))
                )
                # Cast to double safely
                .withColumn(col, F.col(col).cast("double"))
            )

            logger.info(f"Cleaned and casted column: {col}")

        return df_cleaned
