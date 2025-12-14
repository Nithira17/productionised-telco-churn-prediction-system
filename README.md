# Productionised Telco Churn Prediction System

A comprehensive machine learning system for predicting customer churn in the telecommunications industry. This project implements an end-to-end MLOps pipeline using Apache Airflow for orchestration, MLflow for experiment tracking, and multiple ML algorithms for churn prediction.

## Features

- **Data Pipeline**: Automated data ingestion, preprocessing, and feature engineering
- **Model Training**: Hyperparameter tuning and training with XGBoost, LightGBM, and CatBoost
- **Model Evaluation**: Comprehensive model comparison and performance metrics
- **Inference Pipeline**: Real-time and batch prediction capabilities
- **MLOps Integration**: MLflow tracking, model versioning, and experiment management
- **Orchestration**: Apache Airflow DAGs for automated pipeline execution
- **API Deployment**: FastAPI-based REST API for model serving
- **Monitoring**: Integrated logging and monitoring capabilities

## Technologies Used

- **Programming Language**: Python 3.8+
- **Machine Learning**: XGBoost, LightGBM, CatBoost, scikit-learn
- **Data Processing**: Pandas, NumPy, PySpark
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Orchestration**: Apache Airflow
- **Experiment Tracking**: MLflow
- **API Framework**: FastAPI, Uvicorn
- **Configuration**: PyYAML
- **Testing**: pytest
- **Code Quality**: Black, Flake8

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Nithira17/productionised-telco-churn-prediction-system.git
   cd productionised-telco-churn-prediction-system
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure the environment**:
   - Update `config.yaml` with your data paths and configurations
   - Set up environment variables if needed (e.g., for MLflow tracking URI)

## Usage

### Data Preparation
Place your raw telco customer data in `data/raw/` directory. The system expects an Excel file with customer information including demographics, service usage, and churn status.

### Running the Training Pipeline
Execute the training pipeline using Apache Airflow:

```bash
# Start Airflow
airflow webserver -p 8080
airflow scheduler

# Trigger the training DAG
airflow dags trigger train_pipeline_dag
```

### Model Inference
For batch predictions:
```bash
python pipelines/inference_pipeline.py
```


### Monitoring Experiments
Access MLflow UI to track experiments:
```bash
mlflow ui
```

## Project Structure

```
├── airflow_settings.yaml          # Airflow configuration
├── config.yaml                    # Project configuration
├── requirements.txt               # Python dependencies
├── Makefile                       # Build automation
├── artifacts/                     # Model artifacts and data
│   ├── data/                      # Processed datasets
│   ├── models/                    # Trained models
│   └── predictions/               # Prediction outputs
├── dags/                          # Apache Airflow DAGs
│   ├── data_pipeline_dag.py       # Data processing DAG
│   ├── train_pipeline_dag.py      # Model training DAG
│   └── inference_pipeline_dag.py  # Inference DAG
├── data/                          # Data directory
│   ├── raw/                       # Raw input data
│   └── processed/                 # Processed data
├── mlruns/                        # MLflow experiment runs
├── pipelines/                     # Pipeline scripts
│   ├── data_pipeline.py           # Data processing pipeline
│   ├── training_pipeline.py       # Model training pipeline
│   └── streaming_inference_pipeline.py  # Inference pipeline
├── src/                           # Source code
│   ├── data_ingestion.py          # Data loading utilities
│   ├── feature_engineering.py     # Feature engineering modules
│   ├── model_building.py          # Model building classes
│   ├── model_training.py          # Training logic
│   ├── model_evaluation.py        # Evaluation metrics
│   └── model_inference.py         # Inference utilities
└── utils/                         # Utility modules
    ├── config.py                  # Configuration utilities
    ├── mlflow_utils.py            # MLflow helpers
    └── airflow_tasks.py           # Airflow task definitions
```

## Configuration

The system is configured via `config.yaml`. Key configuration sections include:

- **Data Paths**: Define input/output directories and file paths
- **Feature Engineering**: Specify categorical/numerical columns and transformations
- **Model Parameters**: Hyperparameter ranges for different algorithms
- **Preprocessing**: Missing value handling and feature scaling strategies

## Model Performance

The system trains and compares three gradient boosting models:
- XGBoost
- LightGBM  
- CatBoost

Model performance is evaluated using metrics like accuracy, precision, recall, F1-score, and AUC-ROC. The best performing model is automatically selected for deployment.

