import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_project_environment() -> str:
    """Setup project paths and environment variables"""
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    
    paths_to_add = [
        str(project_root),
        str(project_root / 'src'),
        str(project_root / 'utils'),
        str(project_root / 'pipelines')
    ]
    
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    os.environ['PYTHONPATH'] = ':'.join(paths_to_add)
    
    return str(project_root)

def validate_input_data(data_path: str = 'data/raw/telco-customer-dataset.xls') -> Dict[str, Any]:
    """Validate that raw input data exists"""
    project_root = setup_project_environment()
    full_path = Path(project_root) / data_path

    logger.info(f"Validating input data at: {full_path}")

    if not full_path.exists():
        raise FileNotFoundError(f"Input data file not found: {full_path}")
    
    file_size = full_path.stat().st_size
    if file_size == 0:
        raise ValueError(f"Input data file is empty: {full_path}")
    
    logger.info(f"✓ Input data validation passed: {file_size} bytes")
    
    return {
        'status': 'success',
        'file_path': str(full_path),
        'file_size_bytes': file_size
    }

def validate_processed_data() -> Dict[str, Any]:
    """Validate that processed data artifacts exist"""
    project_root = setup_project_environment()
    
    required_files = [
        'artifacts/data/X_train.csv',
        'artifacts/data/X_test.csv',
        'artifacts/data/Y_train.csv',
        'artifacts/data/Y_test.csv'
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = Path(project_root) / file_path
        if not full_path.exists():
            missing_files.append(file_path)
    
    if missing_files:
        raise FileNotFoundError(f"Missing processed data files: {missing_files}")
    
    logger.info("✓ All processed data files validated")
    
    return {
        'status': 'success',
        'message': 'All processed data files exist'
    }

def validate_trained_model() -> Dict[str, Any]:
    """Validate that trained models exist"""
    project_root = setup_project_environment()
    model_dir = Path(project_root) / 'artifacts/models'
    
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Check for model files
    model_files = list(model_dir.glob('*.joblib'))
    
    if not model_files:
        raise FileNotFoundError(f"No trained models found in: {model_dir}")
    
    logger.info(f"✓ Found {len(model_files)} trained model(s)")
    
    return {
        'status': 'success',
        'model_count': len(model_files),
        'models': [m.name for m in model_files]
    }

def run_data_pipeline(
    data_path: str = 'data/raw/telco-customer-dataset.xls',
    force_rebuild: bool = False,
    output_format: str = 'csv'  # Changed from 'both' to avoid Hadoop issues
) -> Dict[str, Any]:
    """Run the data preprocessing pipeline"""
    project_root = setup_project_environment()
    
    try:
        os.chdir(project_root)
        
        from data_pipeline import data_pipeline
        
        logger.info(f"Starting data pipeline: {data_path}")
        
        result = data_pipeline(
            datapath=data_path,
            force_rebuild=force_rebuild,
            output_format=output_format
        )
        
        logger.info("✓ Data pipeline completed successfully")
        
        return {
            'status': 'success',
            'X_train_shape': result['X_train'].shape if 'X_train' in result else None,
            'X_test_shape': result['X_test'].shape if 'X_test' in result else None,
            'message': 'Data pipeline completed'
        }
        
    except Exception as e:
        logger.error(f"✗ Data pipeline failed: {str(e)}")
        raise

def run_training_pipeline(
    train_all_models: bool = True,
    use_hyperparameter_tuning: bool = True
) -> Dict[str, Any]:
    """Run the model training pipeline"""
    project_root = setup_project_environment()
    
    try:
        os.chdir(project_root)
        
        from training_pipeline import train_all_models as train_models
        from config import get_training_config
        
        logger.info("Starting training pipeline")
        
        if train_all_models:
            results, best_model = train_models(
                use_cv=False,
                use_hyperparameter_tuning=use_hyperparameter_tuning
            )
            
            return {
                'status': 'success',
                'best_model': best_model,
                'message': f'Training completed. Best model: {best_model}'
            }
        else:
            from training_pipeline import training_pipeline
            training_config = get_training_config()
            
            model, eval_results = training_pipeline(
                model_type=training_config['default_model_type'],
                use_hyperparameter_tuning=use_hyperparameter_tuning
            )
            
            return {
                'status': 'success',
                'metrics': {k: v for k, v in eval_results.items() if k != 'cm'},
                'message': 'Training completed'
            }
        
    except Exception as e:
        logger.error(f"✗ Training pipeline failed: {str(e)}")
        raise

def run_inference_pipeline(
    sample_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Run inference on sample data"""
    project_root = setup_project_environment()
    
    try:
        os.chdir(project_root)
        
        from streaming_inference_pipeline import get_best_model_path, streaming_inference
        from model_inference import ModelInference
        
        # Get best model
        model_path = get_best_model_path()
        
        # Use default sample if not provided
        if sample_data is None:
            sample_data = {
                "customerID": "TEST-001",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 2,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.7,
                "TotalCharges": 151.65
            }
        
        logger.info(f"Running inference with model: {model_path}")
        
        inference = ModelInference(model_path)
        result = streaming_inference(inference, sample_data)
        
        logger.info("✓ Inference completed successfully")
        
        return {
            'status': 'success',
            'prediction': result,
            'sample_data': sample_data,
            'message': 'Inference completed'
        }
        
    except Exception as e:
        logger.error(f"✗ Inference pipeline failed: {str(e)}")
        raise

def trigger_training_if_needed(**context) -> Dict[str, Any]:
    """Check if model exists, trigger training if needed"""
    try:
        result = validate_trained_model()
        logger.info("✓ Model exists and is valid")
        return {
            'status': 'model_exists',
            'action': 'none'
        }
    except FileNotFoundError:
        logger.warning("⚠️ Model not found, training required")
        
        # In Airflow, you would trigger the training DAG here
        # For now, just return status
        return {
            'status': 'model_missing',
            'action': 'training_required',
            'message': 'Please run training pipeline'
        }