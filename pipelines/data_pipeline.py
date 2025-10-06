import json
import os
import logging
import pandas as pd
from typing import Dict
import numpy as np

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline, PipelineModel
from spark_session import create_spark_session, stop_spark_session
from spark_utils import save_dataframe, spark_to_pandas, get_dataframe_info, check_missing_values

from data_ingestion import DataIngestorCSV
from handle_missing_values import ReplaceValuesStrategy
from feature_engineering import ConvertingToNumeric, NoServiceToNO, CommunicationTypeCreation, TotalInternetServicesCreation
from handle_outliers import IQROutlierDetector, OutlierDetector
from feature_binning import CustomBinningStrategy
from feature_encoding import BinaryFeatureEncodingStrategy, NominalEncodingStrategy, OrdinalEncodingStrategy
from feature_scaling import RobustScalingStrategy
from data_splitter import SimpleTrainTestSplitStrategy
from handle_class_imbalance import SMOTEHandleImbalanceStrategy

from config import get_data_paths, get_columns, get_missing_values_config, get_feature_engineering_config, get_outlier_detection_config, get_feature_binning_config, get_feature_encoding_config, get_feature_scaling_config, get_splitting_config, get_handle_imbalance_config
from mlflow_utils import MLflowTracker, setup_mlflow_autolog, create_mlflow_run_tags
import mlflow

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def log_stage_metrics(df: DataFrame, stage: str, additional_metrics: Dict = None, spark: SparkSession = None):
    try:
        missing_counts = []
        for col in df.columns:
            missing_counts.append(df.filter(F.col(col).isNull()).count())
        total_missing = sum(missing_counts)

        metrics = {
                f'{stage}_rows': df.count(),
                f'{stage}_columns': len(df.columns),
                f'{stage}_missing_values': total_missing,
                f'{stage}_partitions': df.rdd.getNumPartitions()
                }
        
        if additional_metrics:
            metrics.update({f'{stage}_{k}': v for k, v in additional_metrics.items()})
        
        mlflow.log_metrics(metrics)
        logger.info(f"✓ Metrics logged for {stage}: ({metrics[f'{stage}_rows']}, {metrics[f'{stage}_columns']})")
        
    except Exception as e:
        logger.error(f"✗ Failed to log metrics for {stage}: {str(e)}")


def save_processed_data(X_train: DataFrame,
                        X_test: DataFrame,
                        Y_train: DataFrame,
                        Y_test: DataFrame,
                        output_format: str = "both"
                        ) -> Dict[str, str]:
    os.makedirs('artifacts/data', exist_ok=True)
    paths = {}

    if output_format in ["csv", "both"]:
        logger.info("Saving data in CSV format...")

        X_train_pd = spark_to_pandas(X_train)
        X_test_pd = spark_to_pandas(X_test)
        Y_train_pd = spark_to_pandas(Y_train)
        Y_test_pd = spark_to_pandas(Y_test)

        paths['X_train_csv'] = 'artifacts/data/X_train.csv'
        paths['X_test_csv'] = 'artifacts/data/X_test.csv'
        paths['Y_train_csv'] = 'artifacts/data/Y_train.csv'
        paths['Y_test_csv'] = 'artifacts/data/Y_test.csv'
        
        X_train_pd.to_csv(paths['X_train_csv'], index=False)
        X_test_pd.to_csv(paths['X_test_csv'], index=False)
        Y_train_pd.to_csv(paths['Y_train_csv'], index=False)
        Y_test_pd.to_csv(paths['Y_test_csv'], index=False)
        
        logger.info("✓ CSV files saved")

    if output_format in ["parquet", "both"]:
        logger.info("Saving data in Parquet format...")

        paths['X_train_parquet'] = 'artifacts/data/X_train.parquet'
        paths['X_test_parquet'] = 'artifacts/data/X_test.parquet'
        paths['Y_train_parquet'] = 'artifacts/data/Y_train.parquet'
        paths['Y_test_parquet'] = 'artifacts/data/Y_test.parquet'

        # Convert to pandas and save (avoid Spark's Hadoop dependency)
        X_train_pd = spark_to_pandas(X_train)
        X_test_pd = spark_to_pandas(X_test)
        Y_train_pd = spark_to_pandas(Y_train)
        Y_test_pd = spark_to_pandas(Y_test)
        
        X_train_pd.to_parquet(paths['X_train_parquet'], index=False)
        X_test_pd.to_parquet(paths['X_test_parquet'], index=False)
        Y_train_pd.to_parquet(paths['Y_train_parquet'], index=False)
        Y_test_pd.to_parquet(paths['Y_test_parquet'], index=False)
        
        logger.info("✓ Parquet files saved")
    
    return paths



def data_pipeline(
                    datapath: str='data/raw/telco-customer-dataset.xls',
                    target_column: str='Churn',
                    test_size: float=0.2,
                    force_rebuild: bool=False,
                    output_format: str = "both") ->Dict[str, np.ndarray]:
    
    logger.info(f"\n{'='*80}")
    logger.info(f"STARTING PYSPARK DATA PIPELINE")
    logger.info(f"{'='*80}")
    
    if not os.path.exists(datapath):
        logger.error(f"✗ Data file not found: {datapath}")
        raise FileNotFoundError(f"Data file not found: {datapath}")
    
    if not 0 < test_size < 1:
        logger.error(f"✗ Invalid test_size: {test_size}")
        raise ValueError(f"Invalid test_size: {test_size}")
    
    spark = create_spark_session("TelcoChurnPredictionDataPipeline")
    
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
    run_tags = create_mlflow_run_tags('data_pipeline_pyspark', {'data_source': datapath,
                                                        'target_column': target_column,
                                                        'test_size': str(test_size),
                                                        'force_rebuild': str(force_rebuild),
                                                        'output_format': output_format,
                                                        'processing_engine': 'pyspark'})
    run = mlflow_tracker.start_run(run_name='data_pipeline_pyspark', tags=run_tags)

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

    ingestor = DataIngestorCSV(spark)
    df = ingestor.ingest(datapath)

    log_stage_metrics(df, 'raw', spark=spark)

    print('\n-------------------------Step 2: Handle Missing Values---------------------------------')
    value_replacer = ReplaceValuesStrategy(replace_columns=missing_values_config['replace_columns'], spark=spark)
    df = value_replacer.handle(df)
    log_stage_metrics(df, 'missing_handled', spark=spark)

    print('\n-------------------------Step 3: Feature Engineering-----------------------------------')
    numeric_converter = ConvertingToNumeric(convert_numeric_columns=columns['convert_to_numeric'], spark=spark)
    no_service_converter = NoServiceToNO(no_service_columns=feature_engineering_config['no_service_columns'], spark=spark)
    comm_type_creator = CommunicationTypeCreation(comm_type_columns=feature_engineering_config['comm_type_columns'], spark=spark)
    total_internet_services_creator = TotalInternetServicesCreation(internet_services=feature_engineering_config['internet_services'], spark=spark)

    df = numeric_converter.change(df)
    df = no_service_converter.change(df)
    df = comm_type_creator.change(df)
    df = total_internet_services_creator.change(df)
    log_stage_metrics(df, 'feature_engineering_done', spark=spark)

    print('\n---------------------------Step 4: Handling Outliers-------------------------------------')
    initial_count = df.count()
    outlier_detector = OutlierDetector(strategy=IQROutlierDetector(spark=spark))
    df = outlier_detector.handle_outliers(df, selected_columns=columns['numerical_columns'])
    outliers_removed = initial_count - df.count()

    # DROP outlier indicator columns after filtering
    outlier_cols = [f"{col}_outlier" for col in columns['numerical_columns']]
    existing_outlier_cols = [col for col in outlier_cols if col in df.columns]
    if existing_outlier_cols:
        df = df.drop(*existing_outlier_cols)
        logger.info(f"Dropped outlier indicator columns: {existing_outlier_cols}")
        
    log_stage_metrics(df, 'outliers_handled', {'outliers_removed': outliers_removed}, spark=spark)

    print('\n---------------------------Step 5: Feature Binning-------------------------------------')
    custom_binner = CustomBinningStrategy(bin_definitions=binning_config['tenure_bins'], spark=spark)
    df = custom_binner.bin_feature(df, column=columns['binning'][0])
    log_stage_metrics(df, 'feture_binning_done', spark=spark)

    print('\n---------------------------Step 6: Feature Encoding-------------------------------------')
    binary_encoder = BinaryFeatureEncodingStrategy(binary_columns=encoding_config['binary_features'], spark=spark)
    nominal_encoder = NominalEncodingStrategy(nominal_columns=encoding_config['nominal_features'], spark=spark)
    ordinal_encoder = OrdinalEncodingStrategy(ordinal_mappings=encoding_config['ordinal_mappings'], spark=spark)

    df = binary_encoder.encode(df)
    df = nominal_encoder.encode(df)
    df = ordinal_encoder.encode(df)
    log_stage_metrics(df, 'encoded', spark=spark)

    # Save nominal encoders for inference
    logger.info("Saving nominal encoders for inference...")
    for column, encoder_dict in nominal_encoder.get_encoder_dicts().items():
        encoder_path = f'artifacts/encode/{column}_encoder.json'
        with open(encoder_path, 'w') as f:
            json.dump(encoder_dict, f)
    logger.info("Nominal encoders saved")

    print('\n---------------------------Step 7: Feature Scaling-------------------------------------')
    robust_scaling_strategy = RobustScalingStrategy(spark=spark)
    df = robust_scaling_strategy.scale(df, columns_to_scale=scaling_config['columns_to_scale'])
    log_stage_metrics(df, 'scaled', spark=spark)

    print('\n---------------------------Step 8: Post Processing-------------------------------------')
    df = df.drop(*columns['drop_columns'])
    log_stage_metrics(df, 'dropped_columns', spark=spark)

    print('\n---------------------------Step 9: Data Splitting-------------------------------------')
    splitter = SimpleTrainTestSplitStrategy(test_size=splitting_config['test_size'], spark=spark)
    X_train, X_test, Y_train, Y_test = splitter.split(df, target_column="Churn")
    log_stage_metrics(df, 'data_splitted', spark=spark)

    # print('\n---------------------------Step 10: Handling Imbalance-------------------------------------')
    # imbalance_handler = SMOTEHandleImbalanceStrategy(random_state=imbalance_config['random_state'], spark=spark)
    # X_train_resampled, Y_train_resampled = imbalance_handler.handle(X_train.join(Y_train), y_col="Churn")

    print('\n---------------------------Step 10: Handling Imbalance-------------------------------------')
    logger.info("Converting to Pandas for SMOTE (dataset small enough for in-memory processing)")

    # Convert to Pandas early to avoid Spark memory issues
    X_train_pd = spark_to_pandas(X_train)
    Y_train_pd = spark_to_pandas(Y_train)

    logger.info(f"Original training data shape: X={X_train_pd.shape}, Y={Y_train_pd.shape}")

    # Apply SMOTE in Pandas
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=imbalance_config['random_state'])
    X_train_resampled_pd, Y_train_resampled_pd = smote.fit_resample(X_train_pd, Y_train_pd.values.ravel())

    logger.info(f"After SMOTE: X={X_train_resampled_pd.shape}, Y={Y_train_resampled_pd.shape}")

    # Convert back to Spark DataFrames
    X_train_resampled = spark.createDataFrame(pd.DataFrame(X_train_resampled_pd, columns=X_train_pd.columns))
    Y_train_resampled = spark.createDataFrame(pd.DataFrame({'Churn': Y_train_resampled_pd}))

    log_stage_metrics(X_train_resampled, 'imbalance_handled', spark=spark)

    output_paths = save_processed_data(X_train_resampled, X_test, Y_train_resampled, Y_test, output_format=output_format)

    logger.info("✓ Data splitting completed")
    logger.info(f"\nDataset shapes after splitting:")
    logger.info(f"  • X_train: {X_train_resampled.count()} rows, {len(X_train.columns)} columns")
    logger.info(f"  • X_test:  {X_test.count()} rows, {len(X_test.columns)} columns")
    logger.info(f"  • Y_train: {Y_train_resampled.count()} rows, 1 column")
    logger.info(f"  • Y_test:  {Y_test.count()} rows, 1 column")
    logger.info(f"  • Feature columns: {X_train.columns}")


    dataset_info = {
            'total_rows': df.count(),
            'train_rows': X_train_resampled.count(),
            'test_rows': X_test.count(),
            'num_features': len(X_train_resampled.columns),
            'outliers_removed': outliers_removed,
            'test_size': splitting_config['test_size'],
            'random_state': splitting_config.get('random_state', 42),
            'missing_strategy': missing_values_config['strategy'],
            'outlier_method': outlier_config['detection_method'],
            'encoding_applied': True,
            'scaling_applied': True,
            'feature_names': list(X_train_resampled.columns),
            'imbalance_method': imbalance_config['method'],
            'original_train_samples': X_train.count(),
            'resampled_train_samples': X_train_resampled.count(),
            'class_distribution': {
                'train_class_0': Y_train_resampled.filter(F.col('Churn') == 0).count(),
                'train_class_1': Y_train_resampled.filter(F.col('Churn') == 1).count(),
                'test_class_0': Y_test.filter(F.col('Churn') == 0).count(),
                'test_class_1': Y_test.filter(F.col('Churn') == 1).count()
            }
        }
    
    mlflow_tracker.log_data_pipeline_metrics(dataset_info)
    mlflow_tracker.end_run()

    X_train_np = spark_to_pandas(X_train_resampled).values
    X_test_np = spark_to_pandas(X_test).values
    Y_train_np = spark_to_pandas(Y_train_resampled).values.ravel()
    Y_test_np = spark_to_pandas(Y_test).values.ravel()

    stop_spark_session(spark)

    return {
            "X_train": X_train_np,
            "X_test": X_test_np,
            "Y_train": Y_train_np,
            "Y_test": Y_test_np
        }

data_pipeline()