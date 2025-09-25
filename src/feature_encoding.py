import os
import pandas as pd
import json
from sklearn.preprocessing import OneHotEncoder
import logging
from abc import ABC, abstractmethod
logging.basicConfig(level=logging.INFO, format=
                    '%(asctime)s - %(levelname)s - %(message)s')


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
        os.makedirs('artifacts/encode', exist_ok=True)

    def encode(self, df):
        encoded_array = self.encoder.fit_transform(df[self.nominal_columns])
        encoded_feature_names = self.encoder.get_feature_names_out(self.nominal_columns)
        encoded_df = pd.DataFrame(encoded_array, columns=encoded_feature_names, index=df.index)

        encoder_path = os.path.join('artifacts/encode', 'onehot_encoder.json')
        with open(encoder_path, 'w') as f:
            json.dump(
                {
                    "categories": [list(cat) for cat in self.encoder.categories_],
                    "columns": self.nominal_columns
                }, f
            )

        df = df.drop(columns=self.nominal_columns)
        df = pd.concat([df, encoded_df], axis=1)

        return df
    
class OrdinalEncodingStrategy(FeatureEncodingStrategy):
    def __init__(self, ordinal_mappings):
        self.ordinal_mappings = ordinal_mappings

    def encode(self, df):
        for col, mapping in self.ordinal_mappings.items():
            df[col] = df[col].map(mapping)
            logging.info(f'Encoded ordinal variable {col} with {len(mapping)} categories')

        return df