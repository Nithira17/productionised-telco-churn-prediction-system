import pandas as pd
import logging
from abc import ABC, abstractmethod
logging.basicConfig(level=logging.INFO, format=
                    '%(asctime)s - %(levelname)s - %(message)s')

class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def change(self, df: pd.DataFrame) ->pd.DataFrame:
        pass

class ConvertingToNumeric(FeatureEngineeringStrategy):
    def __init__(self, convert_numeric_columns=[]):
        self.convert_numeric_columns = convert_numeric_columns
        logging.info(f'Converting the string values in {self.convert_numeric_columns} to numerical values')

    def change(self, df):
        df[self.convert_numeric_columns[0]] = pd.to_numeric(df[self.convert_numeric_columns[0]], errors='coerce')
        logging.info(f'{self.convert_numeric_columns} was converted into numerical values')
        print(f'\n{df.head(5)}')
        return df
    
class NoServiceToNO(FeatureEngineeringStrategy):
    def __init__(self, no_service_columns=[]):
        self.no_service_columns = no_service_columns
        logging.info(f'Convert No Service values for No in {self.no_service_columns}')

    def change(self, df):
        for col in self.no_service_columns:
            if col == 'MultipleLines':
                df[col] = df[col].replace({'No phone service': 'No'})
            else:
                df[col] = df[col].replace({'No internet service': 'No'})

        logging.info(f'{self.no_service_columns} was updated')
        print(f'\n{df.head(5)}')

        return df
    
class CommunicationTypeCreation(FeatureEngineeringStrategy):
    def __init__(self, comm_type_columns=[]):
        self.comm_type_columns = comm_type_columns
        logging.info(f'Making a new feature, CommunicationType by combining {self.comm_type_columns}')

    def change(self, df):
        def comm_type(row):
            if row['PhoneService'] == 'No' and row['InternetService'] == 'No':
                return 'No Service'
            elif row['PhoneService'] == 'Yes' and row['InternetService'] == 'No':
                return 'Phone Only'
            elif row['PhoneService'] == 'No' and row['InternetService'] != 'No':
                return 'Internet Only'
            else:
                return 'Phone and Internet'

        df['CommunicationType'] = df.apply(comm_type, axis=1)
        logging.info('New Feature CommunicationType has been created')
        print(f'\n{df.head(5)}')

        return df
    
class TotalInternetServicesCreation(FeatureEngineeringStrategy):
    def __init__(self, internet_services=[]):
        self.internet_services = internet_services
        logging.info(f'New feature TotalInternetServices is creating using {self.internet_services}')

    def change(self, df):
        df['TotalInternetServices'] = df[self.internet_services].replace({'Yes': 1, 'No': 0}).sum(axis=1)
        logging.info('New Feature TotalInternetServices has been created')
        print(f'\n{df.head(5)}')

        return df