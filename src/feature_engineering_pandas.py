import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ConvertingToNumeric:
    def __init__(self, convert_numeric_columns):
        self.convert_numeric_columns = convert_numeric_columns
        logger.info(f"Converting the string values in {convert_numeric_columns} to numerical values")

    def change(self, df):
        df_copy = df.copy()
        for col in self.convert_numeric_columns:
            if col in df_copy.columns:
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').fillna(0)
        logger.info(f"{self.convert_numeric_columns} was converted into numerical values")
        return df_copy


class NoServiceToNO:
    def __init__(self, no_service_columns):
        self.no_service_columns = no_service_columns
        logger.info(f"Convert No Service values for No in {no_service_columns}")

    def change(self, df):
        df_copy = df.copy()
        for col in self.no_service_columns:
            if col in df_copy.columns:
                if col == "MultipleLines":
                    df_copy[col] = df_copy[col].replace('No phone service', 'No')
                else:
                    df_copy[col] = df_copy[col].replace('No internet service', 'No')
        logger.info(f"{self.no_service_columns} was updated")
        return df_copy


class CommunicationTypeCreation:
    def __init__(self, comm_type_columns):
        self.comm_type_columns = comm_type_columns
        logger.info(f"Making a new feature, CommunicationType by combining {comm_type_columns}")

    def change(self, df):
        df_copy = df.copy()
        
        def determine_comm_type(row):
            phone = row[self.comm_type_columns[0]]
            internet = row[self.comm_type_columns[1]]
            
            if phone == 'No' and internet == 'No':
                return 'No Service'
            elif phone == 'Yes' and internet == 'No':
                return 'Phone Only'
            elif phone == 'No' and internet != 'No':
                return 'Internet Only'
            else:
                return 'Phone and Internet'
        
        df_copy['CommunicationType'] = df_copy.apply(determine_comm_type, axis=1)
        logger.info("New Feature CommunicationType has been created")
        return df_copy


class TotalInternetServicesCreation:
    def __init__(self, internet_services):
        self.internet_services = internet_services
        logger.info(f"New feature TotalInternetServices is creating using {internet_services}")

    def change(self, df):
        df_copy = df.copy()
        
        def count_services(row):
            count = 0
            for service in self.internet_services:
                if service in row.index and row[service] == 'Yes':
                    count += 1
            return count
        
        df_copy['TotalInternetServices'] = df_copy.apply(count_services, axis=1)
        logger.info("New Feature TotalInternetServices has been created")
        return df_copy