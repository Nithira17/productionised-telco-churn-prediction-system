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

from airflow_tasks import validate_input_data, run_data_pipeline

ist = pytz.timezone('Asia/Colombo')

default_args = {
    'owner': 'Nitira17',
    'depends_on_past': False,
    'start_date': ist.localize(datetime(2025, 10, 6, 11, 30)),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=3),
}

with DAG(
    dag_id='data_pipeline_dag',
    schedule_interval='*/6 * * * *',  # Every 6 minutes
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    description='Data preprocessing pipeline - runs every 6 hours',
    tags=['telco-churn', 'data-pipeline', 'pyspark']
) as dag:
    
    validate_input = PythonOperator(
        task_id='validate_input_data',
        python_callable=validate_input_data,
        execution_timeout=timedelta(minutes=3)
    )

    run_pipeline = PythonOperator(
        task_id='run_data_pipeline',
        python_callable=run_data_pipeline,
        op_kwargs={
            'force_rebuild': False,
            'output_format': 'csv'
        },
        execution_timeout=timedelta(minutes=20)
    )

    validate_input >> run_pipeline