import os
import pandas as pd
import logging
import json

from model_inference import ModelInference
from config import get_model_config, get_training_config, get_inference_config

logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_best_model_path():
    comparison_path = 'artifacts/evaluation/model_comparison.csv'
    
    if os.path.exists(comparison_path):
        comparison_df = pd.read_csv(comparison_path)
        best_model_name = comparison_df.iloc[0]['Model']
        model_path = f'artifacts/models/{best_model_name.lower()}_best_model.joblib'
        logger.info(f"Using best model: {best_model_name} from comparison results")
        return model_path
    else:
        default_model = 'lightgbm_best_model.joblib'
        logger.warning(f"No comparison file found. Using default model: {default_model}")
        return f'artifacts/models/{default_model}'
    
def streaming_inference(inference, data):
    pred = inference.predict(data)
    return pred

# def batch_inference(inference, data_list):
#     results = []
#     for idx, data in enumerate(data_list):
#         logger.info(f"Processing record {idx + 1}/{len(data_list)}")
#         pred = inference.predict(data)
#         pred['customer_id'] = data.get('customerID', f'customer_{idx}')
#         results.append(pred)
#     return results

if __name__ == '__main__':
    # Get the best model path
    model_path = get_best_model_path()
    
    # Initialize inference with the best model
    try:
        inference = ModelInference(model_path)
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        logger.info("Trying alternative models...")
        
        # Try other models if best model fails
        for alt_model in ['catboost_best_model.joblib', 'xgboost_best_model.joblib', 'lightgbm_best_model.joblib']:
            alt_path = f'artifacts/models/{alt_model}'
            if os.path.exists(alt_path):
                try:
                    inference = ModelInference(alt_path)
                    logger.info(f"Successfully loaded alternative model: {alt_model}")
                    break
                except:
                    continue
        else:
            raise ValueError("No valid model found in artifacts/models/")
    
    # Sample data for testing
    data = {
        "customerID": "9237-HQITU",
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
    
    # Single prediction
    logger.info("="*80)
    logger.info("SINGLE PREDICTION TEST")
    logger.info("="*80)
    pred = streaming_inference(inference, data)
    print("\n" + "="*50)
    print("PREDICTION RESULT")
    print("="*50)
    print(json.dumps(pred, indent=2))
    print("="*50 + "\n")
    
    # # Example: Batch prediction
    # logger.info("="*80)
    # logger.info("BATCH PREDICTION TEST")
    # logger.info("="*80)
    
    # batch_data = [
    #     data,
    #     {
    #         "customerID": "test-001",
    #         "gender": "Male",
    #         "SeniorCitizen": 0,
    #         "Partner": "Yes",
    #         "Dependents": "Yes",
    #         "tenure": 12,
    #         "PhoneService": "Yes",
    #         "MultipleLines": "Yes",
    #         "InternetService": "Fiber optic",
    #         "OnlineSecurity": "No",
    #         "OnlineBackup": "No",
    #         "DeviceProtection": "No",
    #         "TechSupport": "No",
    #         "StreamingTV": "Yes",
    #         "StreamingMovies": "Yes",
    #         "Contract": "Month-to-month",
    #         "PaperlessBilling": "Yes",
    #         "PaymentMethod": "Electronic check",
    #         "MonthlyCharges": 89.95,
    #         "TotalCharges": 1079.4
    #     }
    # ]
    
    # batch_results = batch_inference(inference, batch_data)
    # print("\n" + "="*50)
    # print("BATCH PREDICTION RESULTS")
    # print("="*50)
    # for result in batch_results:
    #     print(json.dumps(result, indent=2))
    #     print("-"*50)
