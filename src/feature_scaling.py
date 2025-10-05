import os
from typing import Optional
import joblib
import pandas as pd
import logging
from abc import ABC, abstractmethod
from sklearn.preprocessing import PowerTransformer

from pyspark.ml.feature import MinMaxScaler, StandardScaler, VectorAssembler, RobustScaler
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.functions import vector_to_array
from spark_session import get_or_create_spark_session

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureScalingStrategy(ABC):
    def __init__(self, spark: Optional[SparkSession] = None):
        self.spark = spark or get_or_create_spark_session()
        self.fitted_model = None

    @abstractmethod
    def scale(self, df, columns_to_scale):
        pass

class RobustScalingStrategy(FeatureScalingStrategy):
    def __init__(self, output_col_suffix = "_scaled", spark = None):
        super().__init__(spark)
        self.output_col_suffix = output_col_suffix
        self.scaler_models = {}
        logger.info("RobustScalingStrategy initialized (PySpark)")

    def scale(self, df, columns_to_scale):
        df_scaled = df

        for col in columns_to_scale:
            vector_col = f"{col}_vec"
            df_scaled = df_scaled.withColumn(col, F.col(col).cast("double"))
            assembler = VectorAssembler(inputCols=[col], outputCol=vector_col)

            scaled_vector_col = f"{col}_scaled_vec"
            scaler = RobustScaler(inputCol=vector_col, outputCol=scaled_vector_col)

            pipeline = Pipeline(stages=[assembler, scaler])
            pipeline_model = pipeline.fit(df_scaled)

            df_scaled = pipeline_model.transform(df_scaled)

            df_scaled = df_scaled.withColumn(col,
                                             vector_to_array(F.col(scaled_vector_col)).getItem(0)
                                             )
            
            df_scaled = df_scaled.drop(vector_col, scaled_vector_col)

            self.scaler_models[col] = pipeline_model.stages[-1]

        return df_scaled