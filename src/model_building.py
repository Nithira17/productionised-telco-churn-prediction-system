from abc import ABC, abstractmethod
import joblib
import os
from datetime import datetime
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

class BaseModelBuilder(ABC):
    def __init__(self, model_name:str, **kwargs):
        self.model_name = model_name
        self.model = None
        self.model_params = kwargs

    @abstractmethod
    def build_model(self):
        pass

    def save_model(self, filepath):
        if self.model is None:
            raise ValueError("No model to save. Build the model first")
        
        joblib.dump(self.model, filepath)

    def load_model(self, filepath):
        if not os.path.exists(filepath):
            raise ValueError("Can't load. File not found")
        
        self.model = joblib.load(filepath)

class XGboostModelBuilder(BaseModelBuilder):
    def __init__(self, **kwargs):
        default_params = {'max_depth': 10,
                          'n_estimators': 100,
                          'random_state': 42}
        default_params.update(kwargs)
        super().__init__('XGboost', **default_params)

    def build_model(self):
        self.model = XGBClassifier(**self.model_params)
        return self.model
    
class LightgbmModelBuilder(BaseModelBuilder):
    def __init__(self, **kwargs):
        default_params = {'max_depth': 10,
                          'n_estimators': 100,
                          'random_state': 42}
        default_params.update(kwargs)
        super().__init__('LightGBM', **default_params)

    def build_model(self):
        self.model = LGBMClassifier(**self.model_params)
        return self.model
    
class CatBoostModelBuilder(BaseModelBuilder):
    def __init__(self, **kwargs):
        default_params = {'iterations': 100,
                          'depth': 5}
        default_params.update(kwargs)
        super().__init__('CatBoost', **default_params)

    def build_model(self):
        self.model = CatBoostClassifier(**self.model_params)
        return self.model