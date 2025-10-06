import os
import pandas as pd
import json
import joblib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BinaryFeatureEncodingStrategy:
    def __init__(self, binary_columns):
        self.binary_columns = binary_columns
        logger.info(f"BinaryFeatureEncodingStrategy initialized for columns: {binary_columns}")

    def encode(self, df):
        df_copy = df.copy()
        
        for column in self.binary_columns:
            if column in df_copy.columns:
                # Convert to string first to handle any numeric values
                df_copy[column] = df_copy[column].astype(str)
                
                # Map Yes/No to 1/0
                df_copy[column] = df_copy[column].map({
                    'Yes': 1,
                    'No': 0,
                    '1': 1,
                    '0': 0
                })
                
                # Convert to integer
                df_copy[column] = df_copy[column].fillna(0).astype(int)
        
        logger.info(f"Binary encoding completed for {len(self.binary_columns)} columns")
        return df_copy


class NominalEncodingStrategy:
    def __init__(self, nominal_columns):
        self.nominal_columns = nominal_columns
        self.encoder_dicts = {}
        os.makedirs('artifacts/encode', exist_ok=True)
        logger.info(f"NominalEncodingStrategy initialized for columns: {nominal_columns}")

    def encode(self, df):
        df_copy = df.copy()
        
        for column in self.nominal_columns:
            if column in df_copy.columns:
                # Try to load saved encoder
                encoder_path = f'artifacts/encode/{column}_encoder.json'
                
                if os.path.exists(encoder_path):
                    with open(encoder_path, 'r') as f:
                        encoder_dict = json.load(f)
                    logger.info(f"Loaded encoder for {column}")
                else:
                    # Create new encoder (for training)
                    unique_values = df_copy[column].unique()
                    encoder_dict = {str(val): idx for idx, val in enumerate(unique_values)}
                    
                    # Save encoder
                    with open(encoder_path, 'w') as f:
                        json.dump(encoder_dict, f)
                    logger.info(f"Created and saved encoder for {column}")
                
                self.encoder_dicts[column] = encoder_dict
                
                # Apply encoding
                df_copy[f'{column}_index'] = df_copy[column].astype(str).map(encoder_dict)
                df_copy[f'{column}_index'] = df_copy[f'{column}_index'].fillna(-1).astype(int)
        
        # Drop original nominal columns
        columns_to_drop = [col for col in self.nominal_columns if col in df_copy.columns]
        if columns_to_drop:
            df_copy = df_copy.drop(columns=columns_to_drop)
            logger.info(f"Dropped original nominal columns: {columns_to_drop}")
        
        return df_copy
    
    def load_encoder(self):
        """Load saved encoders from disk"""
        try:
            for column in self.nominal_columns:
                encoder_path = f'artifacts/encode/{column}_encoder.json'
                if os.path.exists(encoder_path):
                    with open(encoder_path, 'r') as f:
                        self.encoder_dicts[column] = json.load(f)
            logger.info("Loaded all encoders successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load encoders: {e}")
            return False


class OrdinalEncodingStrategy:
    def __init__(self, ordinal_mappings):
        self.ordinal_mappings = ordinal_mappings
        logger.info(f"OrdinalEncodingStrategy initialized for columns: {list(ordinal_mappings.keys())}")

    def encode(self, df):
        df_copy = df.copy()
        
        for column, mapping in self.ordinal_mappings.items():
            if column in df_copy.columns:
                df_copy[column] = df_copy[column].map(mapping)
                df_copy[column] = df_copy[column].fillna(-1).astype(int)
                logger.info(f"Ordinal encoding applied to {column}")
        
        return df_copy