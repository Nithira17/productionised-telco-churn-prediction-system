import os
import sys
import logging
import pandas as pd

from data_pipeline import data_pipeline
from typing import Dict, Any, Tuple, Optional

from model_building import XGboostModelBuilder, LightgbmModelBuilder, CatBoostModelBuilder
from model_training import ModelTrainer
from model_evaluation import ModelEvaluator

from config import get_data_paths, get_model_config
from mlflow_utils import MLflowTracker, setup_mlflow_autolog, create_mlflow_run_tags
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def training_pipeline(data_path = 'data/raw/telco-customer-dataset.xls',
                      model_params = None,
                      test_size = 0.2,
                      random_state = 42,
                      model_path = 'artifacts/models/xgboost_cv_model.joblib'):
    if (not os.path.exists(get_data_paths()['X_train'])) or \
       (not os.path.exists(get_data_paths()['X_test'])) or \
       (not os.path.exists(get_data_paths()['Y_train'])) or \
       (not os.path.exists(get_data_paths()['Y_test'])):
        
        data_pipeline()
    else:
        print("Loading Data Artifacts from Data Pipeline")

    mlflow_tracker = MLflowTracker()
    setup_mlflow_autolog()
    run_tags = create_mlflow_run_tags('training_pipeline', {
                                                        'model_type' : 'XGboost',
                                                        'training_stratergy' : 'simple',
                                                        'other_models' : 'lightgbm'
                                                    })
    run = mlflow_tracker.start_run(run_name='training_pipeline', tags=run_tags)

    X_train = pd.read_csv(get_data_paths()['X_train'])
    X_test = pd.read_csv(get_data_paths()['X_test'])
    Y_train = pd.read_csv(get_data_paths()['Y_train'])
    Y_test = pd.read_csv(get_data_paths()['Y_test'])

    model_builder = XGboostModelBuilder(**model_params)
    model = model_builder.build_model()

    trainer = ModelTrainer()
    model, train_score = trainer.train(model=model,
                                       X_train=X_train,
                                       Y_train=Y_train.squeeze())
    trainer.save_model(model, model_path)

    evaluator = ModelEvaluator(model, 'XGboost')
    evaluation_results = evaluator.evaluate(X_test, Y_test)
    evaluation_results_cp = evaluation_results.copy()
    del evaluation_results_cp['cm']

    model_params = get_model_config()['model_params']
    mlflow_tracker.log_training_metrics(model, evaluation_results_cp, model_params)
    mlflow_tracker.end_run()

if __name__ == '__main__':
    model_config = get_model_config()
    training_pipeline(model_params=model_config.get('model_params'))