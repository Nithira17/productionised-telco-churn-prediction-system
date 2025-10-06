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

from airflow_tasks import validate_trained_model, run_inference_pipeline

ist = pytz.timezone('Asia/Colombo')

default_args = {
    'owner': 'zuu-crew',
    'depends_on_past': False,
    'start_date': ist.localize(datetime(2025, 10, 6, 11, 30)),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=2),
}

with DAG(
    dag_id='inference_pipeline_dag',
    schedule_interval='*/30 * * * *',  # Every 30 minutes (not every minute!)
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    description='Inference pipeline - runs every 30 minutes',
    tags=['telco-churn', 'inference', 'prediction']
) as dag:
    
    validate_model = PythonOperator(
        task_id='validate_trained_model',
        python_callable=validate_trained_model,
        execution_timeout=timedelta(minutes=2)
    )

    run_inference = PythonOperator(
        task_id='run_inference_pipeline',  # ✓ FIXED
        python_callable=run_inference_pipeline,
        execution_timeout=timedelta(minutes=5)
    )

    validate_model >> run_inference