import json
import logging
import os
import joblib
import pandas as pd
from feature_encoding import BinaryFeatureEncodingStrategy, NominalEncodingStrategy, OrdinalEncodingStrategy
from feature_engineering import ConvertingToNumeric, NoServiceToNO, CommunicationTypeCreation, TotalInternetServicesCreation
from feature_binning import CustomBinningStrategy
from config import get_columns, get_feature_binning_config, get_feature_encoding_config

logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelInference:
    def __init__(self, model_path):
        self.model_path = model_path
        self.encoders = {}
        self.load_model()
        self.binning_config = get_feature_binning_config()
        self.encoding_config = get_feature_encoding_config()

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise ValueError("Can't load. Model not found")
        
        self.model = joblib.load(self.model_path)

    # def load_encoders(self, encoders_dir):
    #     for file in os.listdir(encoders_dir):
    #         feature_name = file.split

    def preprocess_input(self, data):
        columns = get_columns()
        feature_engineering_config = get_feature_binning_config()
        binning_config = get_feature_binning_config()
        encoding_config = get_feature_encoding_config()
        data = pd.DataFrame([data])

        numeric_converter = ConvertingToNumeric(convert_numeric_columns=columns['convert_to_numeric'])
        no_service_converter = NoServiceToNO(no_service_columns=feature_engineering_config['no_service_columns'])
        comm_type_creator = CommunicationTypeCreation(comm_type_columns=feature_engineering_config['comm_type_columns'])
        total_internet_services_creator = TotalInternetServicesCreation(internet_services=feature_engineering_config['internet_services'])

        data = numeric_converter.change(data)
        data = no_service_converter.change(data)
        data = comm_type_creator.change(data)
        data = total_internet_services_creator.change(data)

        custom_binner = CustomBinningStrategy(bin_definitions=binning_config['tenure_bins'])
        data = custom_binner.bin_feature(data, column=columns['binning'][0])

        binary_encoder = BinaryFeatureEncodingStrategy(binary_columns=encoding_config['binary_features'])
        nominal_encoder = NominalEncodingStrategy(nominal_columns=encoding_config['nominal_features'])
        ordinal_encoder = OrdinalEncodingStrategy(ordinal_mappings=encoding_config['ordinal_mappings'])

        data = binary_encoder.encode(data)
        data = nominal_encoder.encode(data)
        data = ordinal_encoder.encode(data)

        data = data.drop(columns=columns['drop_columns'])

        return data
    
    def predict(self, data):
        pp_data = self.preprocess_input(data)
        Y_pred = self.model.predict(pp_data)
        Y_proba = float(self.model.predict_proba(pp_data)[:, 1])

        Y_pred = 'Churn' if Y_pred==1 else 'Retain'
        Y_proba = round(Y_proba * 100, 2)

        return {"Status": {Y_pred},
                "Confidence": f"{Y_proba}%"}


