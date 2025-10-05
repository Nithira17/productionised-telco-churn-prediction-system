from typing import Optional
import pandas as pd
import logging
from abc import ABC, abstractmethod

from spark_session import get_or_create_spark_session
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

logging.basicConfig(level=logging.INFO, format=
                    '%(asctime)s - %(levelname)s - %(message)s')

class FeatureEngineeringStrategy(ABC):
    def __init__(self, spark: Optional[SparkSession] = None):
        self.spark = spark or get_or_create_spark_session()

    @abstractmethod
    def change(self, df):
        pass

class ConvertingToNumeric(FeatureEngineeringStrategy):
    def __init__(self, convert_numeric_columns=[], spark = None):
        super().__init__(spark)
        self.convert_numeric_columns = convert_numeric_columns
        logging.info(f'Converting the string values in {self.convert_numeric_columns} to numerical values')

    def change(self, df):
        df_converted = df
        for col in self.convert_numeric_columns:
            df_converted = df_converted.withColumn(col,
                                                   F.col(col).cast("double"))
        logging.info(f'{self.convert_numeric_columns} was converted into numerical values')
        
        # df_converted.show(5)
        return df_converted

class NoServiceToNO(FeatureEngineeringStrategy):
    def __init__(self, no_service_columns=[], spark = None):
        super().__init__(spark)
        self.no_service_columns = no_service_columns
        logging.info(f'Convert No Service values for No in {self.no_service_columns}')

    def change(self, df):
        df_changed = df

        for col in self.no_service_columns:
            if col == "MultipleLines":
                df_changed = df_changed.withColumn(col,
                                                   F.when(F.col(col) == 'No phone service', 'No').otherwise(F.col(col))
                                                   )
            else:
                df_changed = df_changed.withColumn(col,
                                                   F.when(F.col(col) == 'No internet service', 'No').otherwise(F.col(col))
                                                   )
                
        logging.info(f'{self.no_service_columns} was updated')
        # df_changed.show(5)
        return df_changed

    
class CommunicationTypeCreation(FeatureEngineeringStrategy):
    def __init__(self, comm_type_columns=[], spark = None):
        super().__init__(spark)
        self.comm_type_columns = comm_type_columns
        logging.info(f'Making a new feature, CommunicationType by combining {self.comm_type_columns}')

    def change(self, df):
        df_changed = df

        df_changed = df_changed.withColumn("CommunicationType",
                                           F.when((F.col('PhoneService') == 'No') & (F.col('InternetService') == 'No'), 'No Service')
                                           .when((F.col('PhoneService') == 'Yes') & (F.col('InternetService') == 'No'), 'Phone Only')
                                           .when((F.col('PhoneService') == 'No') & (F.col('InternetService') != 'No'), 'Internet Only')
                                           .otherwise('Phone and Internet')
                                           )
        
        logging.info('New Feature CommunicationType has been created')
        # df_changed.show(5)

        return df_changed

    
class TotalInternetServicesCreation(FeatureEngineeringStrategy):
    def __init__(self, internet_services=[], spark = None):
        super().__init__(spark)
        self.internet_services = internet_services
        logging.info(f'New feature TotalInternetServices is creating using {self.internet_services}')

    def change(self, df):
        df_changed = df

        for col in self.internet_services:
                df_changed = df_changed.withColumn(col,
                                                   F.when(F.col(col) == 'Yes', F.lit(1)).otherwise(F.lit(0))
                                                   )
                
        df_changed = df_changed.withColumn('TotalInternetServices',
                                           sum(F.col(c) for c in self.internet_services))

        logging.info('New Feature TotalInternetServices has been created')
        # df_changed.show(5)

        return df_changed