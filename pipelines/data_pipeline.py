import os
import pandas as pd
from typing import Dict
import numpy as np
from data_ingestion import DataIngestorCSV
from handle_missing_values import ReplaceValuesStrategy
from feature_engineering import ConvertingToNumeric, NoServiceToNO, CommunicationTypeCreation, TotalInternetServicesCreation
from handle_outliers import IQROutlierDetector, OutlierDetector
from feature_binning import CustomBinningStrategy
from feature_encoding import BinaryFeatureEncodingStrategy, NominalEncodingStrategy, OrdinalEncodingStrategy
from feature_scaling import PowerTransformerScalingStrategy
from data_splitter import SimpleTrainTestSplitStrategy
from handle_class_imbalance import SMOTEHandleImbalanceStrategy
from config import get_data_paths, get_columns, get_missing_values_config, get_feature_engineering_config, get_outlier_detection_config, get_feature_binning_config, get_feature_encoding_config, get_feature_scaling_config, get_splitting_config, get_handle_imbalance_config
from mlflow_utils import MLflowTracker, setup_mlflow_autolog, create_mlflow_run_tags

def data_pipeline(
                    datapath: str='data/raw/telco-customer-dataset.xls',
                    target_column: str='Churn',
                    test_size: float=0.2,
                    force_rebuild: bool=False) ->Dict[str, np.ndarray]:
    
    data_paths = get_data_paths()
    columns = get_columns()
    missing_values_config = get_missing_values_config()
    feature_engineering_config = get_feature_engineering_config()
    outlier_config = get_outlier_detection_config()
    binning_config = get_feature_binning_config()
    encoding_config = get_feature_encoding_config()
    scaling_config = get_feature_scaling_config()
    splitting_config = get_splitting_config()
    imbalance_config = get_handle_imbalance_config()

    mlflow_tracker = MLflowTracker()
    setup_mlflow_autolog()
    run_tags = create_mlflow_run_tags('data_pipeline', {'data_source': datapath,
                                                        'target_column': target_column,
                                                        'test_size': str(test_size),
                                                        'force_rebuild': str(force_rebuild)})
    run = mlflow_tracker.start_run(run_name='data_pipeline', tags=run_tags)

    print('------------------------------Step 1: Data Ingestion----------------------------')
    artifacts_dir = os.path.join(os.path.dirname(__file__), '..', data_paths['data_artifacts_dir'])
    x_train_path = os.path.join(artifacts_dir, 'X_train.csv')
    x_test_path = os.path.join(artifacts_dir, 'X_test.csv')
    y_train_path = os.path.join(artifacts_dir, 'Y_train.csv')
    y_test_path = os.path.join(artifacts_dir, 'Y_test.csv')

    if (not force_rebuild and \
       os.path.exists(x_train_path) and \
       os.path.exists(x_test_path) and \
       os.path.exists(y_train_path) and \
       os.path.exists(y_test_path)):
        
        print("Loading cached preprocessed data...")
        X_train = pd.read_csv(x_train_path)
        X_test = pd.read_csv(x_test_path)
        Y_train = pd.read_csv(y_train_path)
        Y_test = pd.read_csv(y_test_path)

        dataset_info = {
                'total_rows': len(X_train) + len(X_test),
                'train_rows': len(X_train),
                'test_rows': len(X_test),
                'num_features': X_train.shape[1],
                'missing_values': 0,  # Assuming preprocessed data has no missing values
                'outliers_removed': 0,  # Unknown for cached data
                'test_size': len(X_test) / (len(X_train) + len(X_test)),
                'random_state': splitting_config.get('random_state', 42),
                'missing_strategy': 'cached_data',
                'outlier_method': 'cached_data',
                'encoding_applied': True,
                'scaling_applied': True,
                'feature_names': list(X_train.columns),
                'data_source': 'cached',
                'cache_paths': {
                    'X_train': x_train_path,
                    'X_test': x_test_path,
                    'Y_train': y_train_path,
                    'Y_test': y_test_path
                }
            }

        mlflow_tracker.log_data_pipeline_metrics(dataset_info)
        print(f"Loaded cached data - X_train: {X_train.shape}, X_test: {X_test.shape}")

        return {"X_train": X_train,
                "X_test": X_test,
                "Y_train": Y_train,
                "Y_test": Y_test}
    
    os.makedirs(data_paths['data_artifacts_dir'], exist_ok=True)

    ingestor = DataIngestorCSV()
    df = ingestor.ingest(datapath)
    print(f'Loaded data shape: {df.shape}')
    original_shape = df.shape

    print('\n-------------------------Step 2: Handle Missing Values---------------------------------')
    missing_before = df.isnull().sum().sum()
    value_replacer = ReplaceValuesStrategy(replace_columns=missing_values_config['replace_columns'])
    df = value_replacer.handle(df)
    missing_after = df.isnull().sum().sum()
    print(f'Data shape after replacing values: {df.shape}')
    print(f'Missing values before: {missing_before}, after: {missing_after}')

    print('\n-------------------------Step 3: Feature Engineering-----------------------------------')
    numeric_converter = ConvertingToNumeric(convert_numeric_columns=columns['convert_to_numeric'])
    no_service_converter = NoServiceToNO(no_service_columns=feature_engineering_config['no_service_columns'])
    comm_type_creator = CommunicationTypeCreation(comm_type_columns=feature_engineering_config['comm_type_columns'])
    total_internet_services_creator = TotalInternetServicesCreation(internet_services=feature_engineering_config['internet_services'])

    df = numeric_converter.change(df)
    df = no_service_converter.change(df)
    df = comm_type_creator.change(df)
    df = total_internet_services_creator.change(df)
    print(f'Data shape after feature engineering: {df.shape}')
    feature_engineering_shape = df.shape

    print('\n---------------------------Step 4: Handling Outliers-------------------------------------')
    outliers_before = df.shape[0]
    outlier_detector = OutlierDetector(strategy=IQROutlierDetector())
    df = outlier_detector.handle_outliers(df, selected_columns=columns['numerical_columns'])
    outliers_removed = outliers_before - df.shape[0]
    print(f'Data shape after handling outliers: {df.shape}')
    print(f'Outliers removed: {outliers_removed}')

    print('\n---------------------------Step 5: Feature Binning-------------------------------------')
    custom_binner = CustomBinningStrategy(bin_definitions=binning_config['tenure_bins'])
    df = custom_binner.bin_feature(df, column=columns['binning'][0])
    print(f'Data after feature binning: \n{df.head(5)}')

    print('\n---------------------------Step 6: Feature Encoding-------------------------------------')
    binary_encoder = BinaryFeatureEncodingStrategy(binary_columns=encoding_config['binary_features'])
    nominal_encoder = NominalEncodingStrategy(nominal_columns=encoding_config['nominal_features'])
    ordinal_encoder = OrdinalEncodingStrategy(ordinal_mappings=encoding_config['ordinal_mappings'])

    df = binary_encoder.encode(df)
    df = nominal_encoder.encode(df)
    df = ordinal_encoder.encode(df)
    print(f'Data after feature encoding: \n{df.head(5)}')
    print(f'Data shape after feature encoding: {df.shape}')

    print('\n---------------------------Step 7: Feature Scaling-------------------------------------')
    power_transforming_strategy = PowerTransformerScalingStrategy()
    df = power_transforming_strategy.scale(df, columns_to_scale=scaling_config['columns_to_scale'])
    print(f'Data after feature scaling: \n{df.head(5)}')

    print('\n---------------------------Step 8: Post Processing-------------------------------------')
    df = df.drop(columns=columns['drop_columns'])
    print(f'Data after post processing: \n{df.head(5)}')
    print(f'Data shape after post processing: {df.shape}')

    print('\n---------------------------Step 9: Data Splitting-------------------------------------')
    splitter = SimpleTrainTestSplitStrategy(test_size=splitting_config['test_size'])
    X_train, X_test, Y_train, Y_test = splitter.split(df, target_column="Churn")
    print('Data splitting done')

    print('\n---------------------------Step 10: Handling Imbalance-------------------------------------')
    original_train_size = X_train.shape[0]
    imbalance_handler = SMOTEHandleImbalanceStrategy(random_state=imbalance_config['random_state'])
    X_train_resampled, Y_train_resampled = imbalance_handler.handle(X_train, Y_train)

    print(f'Original training samples: {original_train_size}')
    print(f'After SMOTE training samples: {X_train_resampled.shape[0]}')

    X_train_resampled.to_csv(x_train_path, index=False)
    Y_train_resampled.to_csv(y_train_path, index=False)
    X_test.to_csv(x_test_path, index=False)
    Y_test.to_csv(y_test_path, index=False)
    print(f'X train size: {X_train_resampled.shape}')
    print(f'Y train size: {Y_train_resampled.shape}')
    print(f'X test size: {X_test.shape}')
    print(f'Y test size: {Y_test.shape}')

    dataset_info = {
            'total_rows': original_shape[0],
            'train_rows': X_train_resampled.shape[0],
            'test_rows': X_test.shape[0],
            'num_features': X_train_resampled.shape[1],
            'missing_values': missing_before,
            'outliers_removed': outliers_removed,
            'test_size': splitting_config['test_size'],
            'random_state': splitting_config.get('random_state', 42),
            'missing_strategy': missing_values_config['strategy'],
            'outlier_method': outlier_config['detection_method'],
            'encoding_applied': True,
            'scaling_applied': True,
            'feature_names': list(X_train_resampled.columns),
            'imbalance_method': imbalance_config['method'],
            'original_train_samples': original_train_size,
            'resampled_train_samples': X_train_resampled.shape[0],
            'class_distribution': {
                'train_class_0': int((Y_train_resampled == 0).sum()),
                'train_class_1': int((Y_train_resampled == 1).sum()),
                'test_class_0': int((Y_test == 0).sum()),
                'test_class_1': int((Y_test == 1).sum())
            }
        }
    
    mlflow_tracker.log_data_pipeline_metrics(dataset_info)
    mlflow_tracker.end_run()

    return {
            "X_train": X_train_resampled,
            "X_test": X_test,
            "Y_train": Y_train_resampled,
            "Y_test": Y_test
        }

data_pipeline()