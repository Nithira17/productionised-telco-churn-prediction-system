import json
import logging
import os
import joblib
import pandas as pd
import numpy as np
from feature_encoding_pandas import BinaryFeatureEncodingStrategy, NominalEncodingStrategy, OrdinalEncodingStrategy
from feature_engineering_pandas import ConvertingToNumeric, NoServiceToNO, CommunicationTypeCreation, TotalInternetServicesCreation
from feature_binning_pandas import CustomBinningStrategy
from config import get_columns, get_feature_binning_config, get_feature_encoding_config, get_feature_engineering_config, get_feature_scaling_config

logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelInference:
    def __init__(self, model_path):
        self.model_path = model_path
        self.scaler = None
        self.load_model()
        self.binning_config = get_feature_binning_config()
        self.encoding_config = get_feature_encoding_config()
        self.feature_engineering_config = get_feature_engineering_config()
        self.scaling_config = get_feature_scaling_config()
        self.load_scaler()

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise ValueError(f"Can't load. Model not found at {self.model_path}")
        
        self.model = joblib.load(self.model_path)
        logger.info(f"Model loaded from {self.model_path}")

    def load_scaler(self):
        """Load the fitted scaler from training"""
        scaler_path = 'artifacts/preprocessors/power_transformer.joblib'
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Scaler loaded from {scaler_path}")
        else:
            logger.warning(f"Scaler not found at {scaler_path}. Scaling will be skipped!")
            logger.warning("This may affect prediction accuracy!")

    def preprocess_input(self, data):
        """
        Preprocess input data to match training pipeline transformations
        """
        columns = get_columns()
        
        # Convert dict to DataFrame
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        logger.info(f"Input data shape: {data.shape}")
        
        # Step 1: Feature Engineering
        numeric_converter = ConvertingToNumeric(convert_numeric_columns=columns['convert_to_numeric'])
        no_service_converter = NoServiceToNO(no_service_columns=self.feature_engineering_config['no_service_columns'])
        comm_type_creator = CommunicationTypeCreation(comm_type_columns=self.feature_engineering_config['comm_type_columns'])
        total_internet_services_creator = TotalInternetServicesCreation(internet_services=self.feature_engineering_config['internet_services'])

        data = numeric_converter.change(data)
        data = no_service_converter.change(data)
        data = comm_type_creator.change(data)
        data = total_internet_services_creator.change(data)

        # Step 2: Feature Binning
        custom_binner = CustomBinningStrategy(bin_definitions=self.binning_config['tenure_bins'])
        data = custom_binner.bin_feature(data, column=columns['binning'][0])

        # Step 3: Feature Encoding
        # Filter out columns that don't exist in inference data (like 'Churn' target)
        binary_cols_to_encode = [col for col in self.encoding_config['binary_features'] if col in data.columns]
        nominal_cols_to_encode = [col for col in self.encoding_config['nominal_features'] if col in data.columns]
        
        binary_encoder = BinaryFeatureEncodingStrategy(binary_columns=binary_cols_to_encode)
        nominal_encoder = NominalEncodingStrategy(nominal_columns=nominal_cols_to_encode)
        
        # Load the fitted encoder for inference
        encoder_loaded = nominal_encoder.load_encoder()
        if not encoder_loaded:
            logger.warning("Could not load saved encoder. Predictions may be inaccurate!")
        
        ordinal_encoder = OrdinalEncodingStrategy(ordinal_mappings=self.encoding_config['ordinal_mappings'])

        data = binary_encoder.encode(data)
        data = nominal_encoder.encode(data)
        data = ordinal_encoder.encode(data)

        # Step 4: Feature Scaling using saved scaler
        cols_to_scale = [col for col in self.scaling_config['columns_to_scale'] if col in data.columns]
        if cols_to_scale and self.scaler is not None:
            try:
                data[cols_to_scale] = self.scaler.transform(data[cols_to_scale])
                logger.info(f"Applied saved scaler to columns: {cols_to_scale}")
            except Exception as e:
                logger.error(f"Error applying scaler: {e}")
                logger.warning("Proceeding without scaling - predictions may be inaccurate!")
        elif not self.scaler:
            logger.warning("No scaler available - skipping scaling step")

        # Step 5: Drop unnecessary columns
        cols_to_drop = [col for col in columns['drop_columns'] if col in data.columns]
        if cols_to_drop:
            data = data.drop(columns=cols_to_drop)

        logger.info(f"Preprocessed data shape: {data.shape}")

        return data
    
    def predict(self, data):
        """
        Make prediction on preprocessed data
        """
        pp_data = self.preprocess_input(data)
        Y_pred = self.model.predict(pp_data)
        Y_proba = float(self.model.predict_proba(pp_data)[:, 1])

        Y_pred_label = 'Churn' if Y_pred[0] == 1 else 'Retain'
        Y_proba_percent = round(Y_proba * 100, 2)

        result = {
            "Status": Y_pred_label,
            "Confidence": f"{Y_proba_percent}%",
            "Churn_Probability": Y_proba_percent,
            "Retain_Probability": round((1 - Y_proba) * 100, 2)
        }

        logger.info(f"Prediction: {result}")

        return result