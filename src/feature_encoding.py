import os
from typing import Optional
import pandas as pd
import json
import joblib
from sklearn.preprocessing import OneHotEncoder
import logging
from abc import ABC, abstractmethod

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer, OneHotEncoder, IndexToString
from spark_session import get_or_create_spark_session

logging.basicConfig(level=logging.INFO, format=
                    '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureEncodingStrategy(ABC):
    def __init__(self, spark: Optional[SparkSession] = None):
        self.spark = spark or get_or_create_spark_session()

    @abstractmethod
    def encode(self, df: pd.DataFrame) ->pd.DataFrame:
        pass

class BinaryFeatureEncodingStrategy(FeatureEncodingStrategy):
    def __init__(self, binary_columns, spark = None):
        super().__init__(spark)
        self.binary_columns = binary_columns
        logger.info(f"BinaryFeatureEncodingStrategy initialized for columns: {binary_columns}")

    def encode(self, df):
        df_encoded = df

        for column in self.binary_columns:
            # First cast to string to handle any numeric values
            df_encoded = df_encoded.withColumn(
                column,
                F.when(F.col(column).cast("string") == 'Yes', F.lit(1))
                .when(F.col(column).cast("string") == 'No', F.lit(0))
                .when(F.col(column).cast("string") == '1', F.lit(1))
                .when(F.col(column).cast("string") == '0', F.lit(0))
                .otherwise(F.lit(None))  # Use None for unexpected values instead of keeping original
            )
            
            df_encoded = df_encoded.withColumn(column, F.col(column).cast("integer"))
            
        return df_encoded
    
class NominalEncodingStrategy(FeatureEncodingStrategy):
    def __init__(self, nominal_columns, one_hot = False, spark = None):
        super().__init__(spark)
        self.nominal_columns = nominal_columns
        self.one_hot = one_hot
        self.encoder_dicts = {}
        self.indexers = {}
        self.encoders = {}
        os.makedirs('artifacts/encode', exist_ok=True)
        logger.info(f"NominalEncodingStrategy initialized for columns: {nominal_columns}")
        logger.info(f"One-hot encoding: {one_hot}")

    def encode(self, df):
        df_encoded = df

        for column in self.nominal_columns:
            indexer = StringIndexer(inputCol=column,
                                    outputCol=f"{column}_index")
            
            indexer_model = indexer.fit(df_encoded)
            self.indexers[column] = indexer_model

            labels = indexer_model.labels
            encoder_dict = {label: idx for idx, label in enumerate(labels)}
            self.encoder_dicts[column] = encoder_dict

            df_encoded = indexer_model.transform(df_encoded)

        # Drop all original nominal columns after encoding
        df_encoded = df_encoded.drop(*self.nominal_columns)
        logger.info(f"Dropped original nominal columns: {self.nominal_columns}")

        return df_encoded
    
    def get_encoder_dicts(self):
        return self.encoder_dicts
    
    def get_indexers(self):
        return self.indexers

    
class OrdinalEncodingStrategy(FeatureEncodingStrategy):
    def __init__(self, ordinal_mappings, spark = None):
        super().__init__(spark)
        self.ordinal_mappings = ordinal_mappings
        logger.info(f"OrdinalEncodingStrategy initialized for columns: {list(ordinal_mappings.keys())}")

    def encode(self, df):
        df_encoded = df

        for column, mapping in self.ordinal_mappings.items():
            mapping_expr = None
            for value, code in mapping.items():
                if mapping_expr is None:
                    mapping_expr = F.when(F.col(column) == value, F.lit(code))
                else:
                    mapping_expr = mapping_expr.when(F.col(column) == value, F.lit(code))

            # if none of the conditions matched, keep original value
            mapping_expr = mapping_expr.otherwise(F.col(column))

            df_encoded = df_encoded.withColumn(column, mapping_expr)

        return df_encoded