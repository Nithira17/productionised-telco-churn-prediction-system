import os
import sys
import logging
import pandas as pd
from scipy.stats import randint, uniform
import mlflow

from data_pipeline import data_pipeline
from typing import Dict, Any, Tuple, Optional

from model_building import XGboostModelBuilder, LightgbmModelBuilder, CatBoostModelBuilder
from model_training import ModelTrainer
from model_evaluation import ModelEvaluator

from config import get_data_paths, get_model_config, get_training_config
from mlflow_utils import MLflowTracker, setup_mlflow_autolog, create_mlflow_run_tags
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_param_distributions(model_type):
    param_distributions = {
        'XGboost': {
            'n_estimators': randint(100, 501),
            'max_depth': randint(3, 8),
            'learning_rate': uniform(0.01, 0.29),
            'subsample': uniform(0.8, 0.2),
            'colsample_bytree': uniform(0.8, 0.2),
            'reg_alpha': uniform(0, 1),
            'reg_lambda': uniform(0, 1)
        },
        'LightGBM': {
            'n_estimators': randint(100, 501),
            'max_depth': randint(3, 8),
            'learning_rate': uniform(0.01, 0.29),
            'subsample': uniform(0.8, 0.2),
            'colsample_bytree': uniform(0.8, 0.2),
            'num_leaves': randint(15, 128),
            'min_child_samples': randint(10, 31),
            'reg_alpha': uniform(0, 1),
            'reg_lambda': uniform(0, 1)
        },
        'CatBoost': {
            'iterations': randint(100, 501),
            'depth': randint(3, 8),
            'learning_rate': uniform(0.01, 0.29),
            'subsample': uniform(0.8, 0.2),
            'colsample_bylevel': uniform(0.8, 0.2),
            'l2_leaf_reg': randint(1, 10),
            'border_count': [32, 64, 128],
            'bagging_temperature': uniform(0, 10)
        }
    }
    return param_distributions.get(model_type, {})

def get_base_model_params(model_type):
    base_params = {
        'XGboost': {
            'random_state': 42,
            'eval_metric': 'logloss'
        },
        'LightGBM': {
            'random_state': 42,
            'verbose': -1
        },
        'CatBoost': {
            'random_state': 42,
            'verbose': False
        }
    }
    return base_params.get(model_type, {'random_state': 42})


def training_pipeline(data_path = 'data/raw/telco-customer-dataset.xls',
                      model_params = None,
                      test_size = 0.2,
                      random_state = 42,
                      model_path = 'artifacts/models/xgboost_cv_model.joblib',
                      use_cv = False,
                      use_hyperparameter_tuning = False,
                      model_type = 'XGboost'):
    
    try:
        mlflow.end_run()
    except:
        pass
    
    if (not os.path.exists(get_data_paths()['X_train'])) or \
       (not os.path.exists(get_data_paths()['X_test'])) or \
       (not os.path.exists(get_data_paths()['Y_train'])) or \
       (not os.path.exists(get_data_paths()['Y_test'])):
        
        data_pipeline()
    else:
        print("Loading Data Artifacts from Data Pipeline")

    mlflow_tracker = MLflowTracker()
    setup_mlflow_autolog()

    training_strategy = 'cv' if use_cv else 'hyperparameter_tuning' if use_hyperparameter_tuning else 'simple'
    run_tags = create_mlflow_run_tags('training_pipeline', {
                                                        'model_type' : model_type,
                                                        'training_strategy' : training_strategy
                                                    })
    run = mlflow_tracker.start_run(run_name=f'training_pipeline_{training_strategy}', tags=run_tags)

    X_train = pd.read_csv(get_data_paths()['X_train'])
    X_test = pd.read_csv(get_data_paths()['X_test'])
    Y_train = pd.read_csv(get_data_paths()['Y_train'])
    Y_test = pd.read_csv(get_data_paths()['Y_test'])

    if use_hyperparameter_tuning:
        model_params = get_base_model_params(model_type)
    
    if model_params is None:
        model_params = get_base_model_params(model_type)

    if model_type == 'XGboost':
        model_builder = XGboostModelBuilder(**model_params)
    elif model_type == 'LightGBM':
        model_builder = LightgbmModelBuilder(**model_params)
    elif model_type == 'CatBoost':
        model_builder = CatBoostModelBuilder(**model_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    base_model = model_builder.build_model()
    trainer = ModelTrainer()

    if use_hyperparameter_tuning:
        logger.info(f"Starting hyperparameter tuning for {model_type}")
        param_distributions = get_param_distributions(model_type)

        model, best_params, best_cv_score = trainer.hyperparameter_search(model=base_model,
                                                                          X_train=X_train,
                                                                          Y_train=Y_train.squeeze(),
                                                                          param_distributions=param_distributions,
                                                                          n_iter=100,
                                                                          cv_folds=5,
                                                                          random_state=random_state)
        mlflow_tracker.log_params(best_params)
        mlflow_tracker.log_metric('best_cv_f1_score', best_cv_score)
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best CV F1 Score: {best_cv_score}")

    elif use_cv:
        logger.info(f"Starting cross-validation training for {model_type}")
        model, cv_results = trainer.train_with_cv(model=base_model,
                                                  X_train=X_train,
                                                  Y_train=Y_train.squeeze(),
                                                  cv_folds=5,
                                                  random_state=random_state)
        
        cv_metrics = {'cv_accuracy_mean': cv_results['test_accuracy'].mean(),
                      'cv_accuracy_std': cv_results['test_accuracy'].std(),
                      'cv_precision_mean': cv_results['test_precision'].mean(),
                      'cv_precision_std': cv_results['test_precision'].std(),
                      'cv_recall_mean': cv_results['test_recall'].mean(),
                      'cv_recall_std': cv_results['test_recall'].std(),
                      'cv_f1_mean': cv_results['test_f1'].mean(),
                      'cv_f1_std': cv_results['test_f1'].std()
                      }
        
        mlflow_tracker.log_metrics(cv_metrics)

        logger.info(f"CV F1 Score: {cv_metrics['cv_f1_mean']:.4f} (+/- {cv_metrics['cv_f1_std']:.4f})")

    else:
        logger.info(f"Starting simple training for {model_type}")
        model, train_score = trainer.train(model=base_model,
                                           X_train=X_train,
                                           Y_train=Y_train.squeeze()
                                           )
        mlflow_tracker.log_metric('train_score', train_score)

    
    trainer.save_model(model, model_path)
    logger.info(f"Model saved to {model_path}")

    evaluator = ModelEvaluator(model, model_type)
    evaluation_results = evaluator.evaluate(X_test, Y_test)
    evaluation_results_cp = evaluation_results.copy()
    del evaluation_results_cp['cm']

    mlflow_tracker.log_training_metrics(model, evaluation_results_cp, model_params)

    mlflow_tracker.log_param('training_strategy', training_strategy)
    mlflow_tracker.log_param('model_type', model_type)

    mlflow_tracker.end_run()

    logger.info(f"Training completed. Test metrics: {evaluation_results_cp}")

    return model, evaluation_results

def train_all_models(use_cv=False, use_hyperparameter_tuning=True):
    model_types = ['XGboost', 'LightGBM', 'CatBoost']
    results = {}
    
    logger.info("="*80)
    logger.info("STARTING MULTI-MODEL TRAINING COMPARISON")
    logger.info("="*80)
    
    for model_type in model_types:
        logger.info(f"\n{'='*80}")
        logger.info(f"Training {model_type} model")
        logger.info(f"{'='*80}\n")
        
        try:
            model_path = f'artifacts/models/{model_type.lower()}_best_model.joblib'
            
            model, evaluation_results = training_pipeline(
                model_type=model_type,
                use_cv=use_cv,
                use_hyperparameter_tuning=use_hyperparameter_tuning,
                model_path=model_path
            )
            
            results[model_type] = {
                'model': model,
                'metrics': evaluation_results,
                'model_path': model_path
            }
            
            logger.info(f"\n{model_type} Results:")
            logger.info(f"  Accuracy:  {evaluation_results.get('accuracy', 0):.4f}")
            logger.info(f"  Precision: {evaluation_results.get('precision', 0):.4f}")
            logger.info(f"  Recall:    {evaluation_results.get('recall', 0):.4f}")
            logger.info(f"  F1 Score:  {evaluation_results.get('f1', 0):.4f}")
            
        except Exception as e:
            logger.error(f"Error training {model_type}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            results[model_type] = {
                'error': str(e)
            }
    
    best_model_name = None
    best_f1_score = -1
    
    logger.info(f"\n{'='*80}")
    logger.info("MODEL COMPARISON SUMMARY")
    logger.info(f"{'='*80}\n")
    
    comparison_df = []
    for model_name, result in results.items():
        if 'error' not in result:
            metrics = result['metrics']
            f1_score = metrics.get('f1', 0)
            
            comparison_df.append({
                'Model': model_name,
                'Accuracy': metrics.get('accuracy', 0),
                'Precision': metrics.get('precision', 0),
                'Recall': metrics.get('recall', 0),
                'F1 Score': f1_score
            })
            
            if f1_score > best_f1_score:
                best_f1_score = f1_score
                best_model_name = model_name
        else:
            logger.warning(f"{model_name} failed with error: {result['error']}")
    
    if not comparison_df:
        logger.error("All models failed to train! Check the errors above.")
        return results, None
    
    comparison_df = pd.DataFrame(comparison_df)
    comparison_df = comparison_df.sort_values('F1 Score', ascending=False)
    print("\n" + comparison_df.to_string(index=False))
    
    if best_model_name:
        logger.info(f"\n{'='*80}")
        logger.info(f"BEST MODEL: {best_model_name}")
        logger.info(f"   F1 Score: {best_f1_score:.4f}")
        logger.info(f"   Model Path: {results[best_model_name]['model_path']}")
        logger.info(f"{'='*80}\n")
        
        os.makedirs('artifacts/evaluation', exist_ok=True)
        comparison_df.to_csv('artifacts/evaluation/model_comparison.csv', index=False)
        logger.info("Model comparison saved to: artifacts/evaluation/model_comparison.csv")
    
    return results, best_model_name


if __name__ == '__main__':
    training_config = get_training_config()
    
    train_all = training_config.get('train_all_models', False)
    
    if train_all:
        use_cv = training_config['default_training_strategy'] == 'cv'
        use_hyperparameter_tuning = training_config.get('hyperparameter_tuning', {}).get('enabled', False)
        
        results, best_model = train_all_models(
            use_cv=use_cv,
            use_hyperparameter_tuning=use_hyperparameter_tuning
        )
    else:
        use_cv = training_config['default_training_strategy'] == 'cv'
        use_hyperparameter_tuning = training_config.get('hyperparameter_tuning', {}).get('enabled')
        model_type = training_config['default_model_type']
        
        training_pipeline(
            model_params=get_base_model_params(model_type),
            use_cv=use_cv,
            use_hyperparameter_tuning=use_hyperparameter_tuning,
            model_type=model_type
        )