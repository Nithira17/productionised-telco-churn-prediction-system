import os
import logging
import mlflow
import mlflow.sklearn
from typing import Dict, Any, Optional, Union
from datetime import datetime
import numpy as np

from config import get_mlflow_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MLflowTracker:
    def __init__(self):
        self.config = get_mlflow_config()
        self.setup_mlflow()

    def setup_mlflow(self):
        tracking_uri = self.config['tracking_uri']
        mlflow.set_tracking_uri(tracking_uri)

        experiment_name = self.config['experiment_name']

        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                logger.info(f"Created new MLflow experiment: {experiment_name} (ID: {experiment_id})")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing MLflow experiment: {experiment_name} (ID: {experiment_id})")
                
            mlflow.set_experiment(experiment_name)
            
        except Exception as e:
            logger.error(f"Error setting up MLflow experiment: {e}")
            raise
    
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> mlflow.ActiveRun:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if run_name is None:
            run_name_prefix = self.config['run_name_prefix']
            run_name_prefix = run_name_prefix.replace('_', ' ')
            run_name = f"{run_name_prefix} | {timestamp}"
        else:
            run_name = run_name.replace('_', ' ')
            run_name = f"{run_name} | {timestamp}"

        default_tags = self.config.get('tags', {})
        if tags:
            default_tags.update(tags)

        run = mlflow.start_run(run_name=run_name, tags=default_tags)
        logger.info(f"Started MLflow run: {run_name} (ID: {run.info.run_id})")
        print(f"🎯 MLflow Run Name: {run_name}")
        return run
    
    def log_data_pipeline_metrics(self, dataset_info: Dict[str, Any]):
        try:
            mlflow.log_metrics({
                'dataset_rows': dataset_info.get('total_rows', 0),
                'training_rows': dataset_info.get('train_rows', 0),
                'test_rows': dataset_info.get('test_rows', 0),
                'num_features': dataset_info.get('num_features', 0),
                'missing_values_count': dataset_info.get('missing_values', 0),
                'outliers_removed': dataset_info.get('outliers_removed', 0)
            })

            mlflow.log_params({
                'test_size': dataset_info.get('test_size', 0.2),
                'random_state': dataset_info.get('random_state', 42),
                'missing_value_strategy': dataset_info.get('missing_strategy', 'unknown'),
                'outlier_detection_method': dataset_info.get('outlier_method', 'unknown'),
                'feature_encoding_applied': dataset_info.get('encoding_applied', False),
                'feature_scaling_applied': dataset_info.get('scaling_applied', False)
            })

            if 'feature_names' in dataset_info:
                mlflow.log_param('feature_names', str(dataset_info['feature_names']))
            
            logger.info("Logged data pipeline metrics to MLflow")

        except Exception as e:
            logger.error(f"Error logging data pipeline metrics: {e}")

    def log_training_metrics(self, model, training_metrics: Dict[str, Any], model_params: Dict[str, Any]):
        try:
            mlflow.log_params(model_params)
            mlflow.log_metrics(training_metrics)

            artifact_path = self.config['artifact_path']
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=artifact_path,
                registered_model_name=self.config['model_registry_name']
            )

            logger.info("Logged training metrics and model to MLflow")
        
        except Exception as e:
            logger.error(f"Error logging training metrics: {e}")

    def log_params(self, params: Dict[str, Any]):
        try:
            mlflow.log_params(params)
            logger.info("Logged parameters to MLflow")
        except Exception as e:
            logger.error(f"Error logging parameters: {e}")

    def log_metrics(self, metrics: Dict[str, float]):
        try:
            mlflow.log_metrics(metrics)
            logger.info("Logged metrics to MLflow")
        except Exception as e:
            logger.error(f"Error logging metrics: {e}")
    
    def log_param(self, key: str, value: Any):
        try:
            mlflow.log_param(key, value)
        except Exception as e:
            logger.error(f"Error logging parameter {key}: {e}")
    
    def log_metric(self, key: str, value: float):
        try:
            mlflow.log_metric(key, value)
        except Exception as e:
            logger.error(f"Error logging metric {key}: {e}")

    def log_evaluation_metrics(self, evaluation_metrics: Dict[str, Any], confusion_matrix_path: Optional[str] = None):
        try:
            if 'metrics' in evaluation_metrics:
                mlflow.log_metrics(evaluation_metrics['metrics'])

            if confusion_matrix_path and os.path.exists(confusion_matrix_path):
                mlflow.log_artifact(confusion_matrix_path, "evaluation")
            
            logger.info("Logged evaluation metrics to MLflow")

        except Exception as e:
            logger.error(f"Error logging evaluation metrics: {e}")

    def log_inference_metrics(self, predictions: np.ndarray, probabilities: Optional[np.ndarray] = None, 
                            input_data_info: Optional[Dict[str, Any]] = None):
        try:
            inference_metrics = {
                'num_predictions': len(predictions),
                'avg_prediction': float(np.mean(predictions)),
                'prediction_distribution_churn': int(np.sum(predictions)),
                'prediction_distribution_retain': int(len(predictions) - np.sum(predictions))
            }

            if probabilities is not None:
                inference_metrics.update({
                    'avg_churn_probability': float(np.mean(probabilities)),
                    'high_risk_predictions': int(np.sum(probabilities > 0.7)),
                    'medium_risk_predictions': int(np.sum((probabilities > 0.5) & (probabilities <= 0.7))),
                    'low_risk_predictions': int(np.sum(probabilities <= 0.5))
                })

            mlflow.log_metrics(inference_metrics)

            if input_data_info:
                mlflow.log_params(input_data_info)
            
            logger.info("Logged inference metrics to MLflow")

        except Exception as e:
            logger.error(f"Error logging inference metrics: {e}")

    def load_model_from_registry(self, model_name: Optional[str] = None, 
                               version: Optional[Union[int, str]] = None, 
                               stage: Optional[str] = None):
        try:
            if model_name is None:
                model_name = self.config['model_registry_name']

            if stage:
                model_uri = f"models:/{model_name}/{stage}"
            elif version:
                model_uri = f"models:/{model_name}/{version}"
            else:
                model_uri = f"models:/{model_name}/latest"
            
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Loaded model from MLflow registry: {model_uri}")
            return model

        except Exception as e:
            logger.error(f"Error loading model from MLflow registry: {e}")
            return None

    def get_latest_model_version(self, model_name: Optional[str] = None) -> Optional[str]:
        try:
            if model_name is None:
                model_name = self.config['model_registry_name']

            client = mlflow.tracking.MlflowClient()
            latest_version = client.get_latest_versions(model_name, stages=["None", "Staging", "Production"])

            if latest_version:
                return latest_version[0].version
            return None

        except Exception as e:
            logger.error(f"Error getting latest model version: {e}")
            return None

    def transition_model_stage(self, model_name: Optional[str] = None, 
                             version: Optional[str] = None, 
                             stage: str = "Staging"):
        try:
            if model_name is None:
                model_name = self.config['model_registry_name']

            if version is None:
                version = self.get_latest_model_version(model_name)

            if version:
                client = mlflow.tracking.MlflowClient()
                client.transition_model_version_stage(
                    name=model_name,
                    version=version,
                    stage=stage
                )
                logger.info(f"Transitioned model {model_name} version {version} to {stage}")

        except Exception as e:
            logger.error(f"Error transitioning model stage: {e}")

    def end_run(self):
        try:
            mlflow.end_run()
            logger.info("Ended MLflow run")
        except Exception as e:
            logger.error(f"Error ending MLflow run: {e}")

def setup_mlflow_autolog():
    mlflow_config = get_mlflow_config()
    if mlflow_config['autolog']:
        mlflow.sklearn.autolog()
        logger.info("MLflow autologging enabled for scikit-learn")

def create_mlflow_run_tags(pipeline_type: str, additional_tags: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    tags = {
        'pipeline_type': pipeline_type,
        'timestamp': datetime.now().isoformat(),
    }
    
    if additional_tags:
        tags.update(additional_tags)
    
    return tags                         