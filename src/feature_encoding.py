import os
import pandas as pd
import json
import joblib
from sklearn.preprocessing import OneHotEncoder
import logging
from abc import ABC, abstractmethod
logging.basicConfig(level=logging.INFO, format=
                    '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureEncodingStrategy(ABC):
    @abstractmethod
    def encode(self, df: pd.DataFrame) ->pd.DataFrame:
        pass

class BinaryFeatureEncodingStrategy(FeatureEncodingStrategy):
    def __init__(self, binary_columns):
        self.binary_columns = binary_columns

    def encode(self, df):
        for col in self.binary_columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})

        return df
    
class NominalEncodingStrategy(FeatureEncodingStrategy):
    def __init__(self, nominal_columns):
        self.nominal_columns = nominal_columns
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.fitted = False
        os.makedirs('artifacts/preprocessors', exist_ok=True)

    def encode(self, df):
        # Fit and transform during training
        if not self.fitted:
            encoded_array = self.encoder.fit_transform(df[self.nominal_columns])
            encoded_feature_names = self.encoder.get_feature_names_out(self.nominal_columns)
            encoded_df = pd.DataFrame(encoded_array, columns=encoded_feature_names, index=df.index)

            # Save encoder
            encoder_path = os.path.join('artifacts/preprocessors', 'onehot_encoder.joblib')
            joblib.dump(self.encoder, encoder_path)
            logger.info(f"Saved OneHotEncoder to {encoder_path}")
            
            # Also save feature names for reference
            encoder_info_path = os.path.join('artifacts/preprocessors', 'onehot_encoder_info.json')
            with open(encoder_info_path, 'w') as f:
                json.dump(
                    {
                        "categories": [list(cat) for cat in self.encoder.categories_],
                        "columns": self.nominal_columns,
                        "feature_names": list(encoded_feature_names)
                    }, f, indent=2
                )
            
            self.fitted = True
        else:
            # Transform only during inference (encoder already fitted)
            encoded_array = self.encoder.transform(df[self.nominal_columns])
            encoded_feature_names = self.encoder.get_feature_names_out(self.nominal_columns)
            encoded_df = pd.DataFrame(encoded_array, columns=encoded_feature_names, index=df.index)

        df = df.drop(columns=self.nominal_columns)
        df = pd.concat([df, encoded_df], axis=1)

        return df
    
    def load_encoder(self):
        """Load saved encoder for inference"""
        encoder_path = os.path.join('artifacts/preprocessors', 'onehot_encoder.joblib')
        if os.path.exists(encoder_path):
            self.encoder = joblib.load(encoder_path)
            self.fitted = True
            logger.info(f"Loaded OneHotEncoder from {encoder_path}")
            return True
        return False
    
class OrdinalEncodingStrategy(FeatureEncodingStrategy):
    def __init__(self, ordinal_mappings):
        self.ordinal_mappings = ordinal_mappings

    def encode(self, df):
        for col, mapping in self.ordinal_mappings.items():
            df[col] = df[col].map(mapping)
            logger.info(f'Encoded ordinal variable {col} with {len(mapping)} categories')

        return df