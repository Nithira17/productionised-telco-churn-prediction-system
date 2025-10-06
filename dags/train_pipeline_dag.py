import os
import sys
from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator
import pytz

project_root = '/mnt/c/Users/hewaj/Desktop/Zuu Crew/Productionised Telco Churn Prediction System'
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'utils'))
sys.path.insert(0, os.path.join(project_root, 'src'))

from airflow_tasks import validate_processed_data, run_training_pipeline

ist = pytz.timezone('Asia/Colombo')

default_args = {
    'owner': 'zuu-crew',
    'depends_on_past': True,  # Training depends on data pipeline
    'start_date': ist.localize(datetime(2025, 10, 6, 11, 30)),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=15),
}

with DAG(
    dag_id='train_pipeline_dag',
    schedule_interval='0 2 * * 0',  # Weekly on Sunday at 2 AM IST
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    description='Model training pipeline - runs weekly',
    tags=['telco-churn', 'training', 'ml']
) as dag:
    
    validate_data = PythonOperator(
        task_id='validate_processed_data',
        python_callable=validate_processed_data,
        execution_timeout=timedelta(minutes=3)
    )

    train_models = PythonOperator(
        task_id='run_training_pipeline',
        python_callable=run_training_pipeline,
        op_kwargs={
            'train_all_models': True,
            'use_hyperparameter_tuning': True
        },
        execution_timeout=timedelta(hours=2)
    )

    validate_data >> train_models