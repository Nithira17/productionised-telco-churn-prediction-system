import os
import joblib
import pandas as pd
import logging
from abc import ABC, abstractmethod
from sklearn.preprocessing import PowerTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureScalingStrategy(ABC):
    @abstractmethod
    def scale(self, df: pd.DataFrame, columns_to_scale: list) -> pd.DataFrame:
        pass

class PowerTransformerScalingStrategy(FeatureScalingStrategy):
    def __init__(self):
        self.scaler = PowerTransformer()
        self.fitted = False

    def scale(self, df, columns_to_scale):
        if not self.fitted:
            # Fit and transform during training
            df[columns_to_scale] = self.scaler.fit_transform(df[columns_to_scale])
            self.fitted = True
            
            # Save the fitted scaler for inference
            os.makedirs('artifacts/preprocessors', exist_ok=True)
            scaler_path = 'artifacts/preprocessors/power_transformer.joblib'
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Saved fitted PowerTransformer to {scaler_path}")
        else:
            # Transform only during inference
            df[columns_to_scale] = self.scaler.transform(df[columns_to_scale])
        
        logger.info(f'Applied PowerTransformer scaling to columns: {columns_to_scale}')
        return df