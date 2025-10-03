import pandas as pd
import logging
from typing import Optional
from abc import ABC, abstractmethod
from pyspark.sql import DataFrame, SparkSession
from spark_session import get_

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataIngestor(ABC):
    def __init__(self, spark: Optional[SparkSession] = None):
    @abstractmethod
    def ingest(self, file_path_or_link: str) ->pd.DataFrame:
        pass

class DataIngestorCSV(DataIngestor):
    def ingest(self, file_path_or_link):
        return pd.read_csv(file_path_or_link)
    
class DataIngestorExcel(DataIngestor):
    def ingest(self, file_path_or_link):
        return pd.read_excel(file_path_or_link)