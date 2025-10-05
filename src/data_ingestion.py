import pandas as pd
import logging
from typing import Optional
from abc import ABC, abstractmethod
from pyspark.sql import DataFrame, SparkSession
from spark_session import get_or_create_spark_session

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataIngestor(ABC):
    def __init__(self, spark: Optional[SparkSession] = None):
        self.spark = spark or get_or_create_spark_session()

    @abstractmethod
    def ingest(self, file_path_or_link: str) ->pd.DataFrame:
        pass

class DataIngestorCSV(DataIngestor):
    def ingest(self, file_path_or_link, **options):
        logger.info(f"\n{'='*60}")
        logger.info(f"DATA INGESTION - CSV (PySpark)")
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting CSV data ingestion from: {file_path_or_link}")

        try:
            csv_options = {
                        "header": "true",
                        "inferSchema": "true",
                        "ignoreLeadingWhiteSpace": "true",
                        "ignoreTrailingWhiteSpace": "true",
                        "nullValue": "",
                        "nanValue": "NaN",
                        "escape": '"',
                        "quote": '"'
                        }
            csv_options.update(options)

            # df = pd.read_csv(file_path_or_link)
            df = self.spark.read.options(**csv_options).csv(file_path_or_link)

        except Exception as e:
            logger.error(f"✗ Failed to load CSV data from {file_path_or_link}: {str(e)}")
            logger.info(f"{'='*60}\n")
            raise

        return df
        
    
class DataIngestorExcel(DataIngestor):
    def ingest(self, file_path_or_link, sheet_name = None, **options):
        logger.info(f"\n{'='*60}")
        logger.info(f"DATA INGESTION - EXCEL (PySpark)")
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting Excel data ingestion from: {file_path_or_link}")

        try:
            logger.info('Note: Using Pandas for excel reading, then converting to pyspark')
            
            # df = pd.read_excel(file_path_or_link)

            pandas_df = pd.read_excel(file_path_or_link)
            df = self.spark.createDataFrame(pandas_df)

        except Exception as e:
            logger.error(f"✗ Failed to load Excel data from {file_path_or_link}: {str(e)}")
            logger.info(f"{'='*60}\n")
            raise

        return df
    
class DataIngestorParquet(DataIngestor):
    def ingest(self, file_path_or_link, **options):
        logger.info(f"\n{'='*60}")
        logger.info(f"DATA INGESTION - Parquet (PySpark)")
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting Parquet data ingestion from: {file_path_or_link}")

        try:
            df = self.spark.read.options(**options).parquet(file_path_or_link)

        except Exception as e:
            logger.error(f"✗ Failed to load Parquet data from {file_path_or_link}: {str(e)}")
            logger.info(f"{'='*60}\n")
            raise

        return df