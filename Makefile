.PHONY: all clean install train-pipeline data-pipeline streaming-inference run-all mlflow-ui help

# Default Python interpreter
PYTHON = python
VENV = .venv/Scripts/activate

# Set PYTHONPATH to include src and utils directories
PYTHONPATH = $(shell cd)\\src;$(shell cd)\\utils

# MLFLOW
MLFLOW_PORT ?= 5000
WORKERS = 1

# Default target
all: help

# Help target
help:
	@echo Available targets:
	@echo   make install             - Install project dependencies and set up environment
	@echo   make data-pipeline       - Run the data pipeline
	@echo   make train-pipeline      - Run the training pipeline
	@echo   make streaming-inference - Run the streaming inference pipeline with the sample JSON
	@echo   make run-all             - Run all pipelines in sequence
	@echo   make mlflow-ui           - Launch the MLflow UI on localhost:$(MLFLOW_PORT)
	@echo   make clean               - Clean up artifacts
	@echo   make sync-dags-to-wsl    - Sync DAG files from Windows to WSL2 Airflow
	@echo   make airflow-start-wsl   - Start Airflow in WSL2
	@echo   make airflow-status      - Check if Airflow is running
	@echo   make airflow-stop-wsl    - Stop Airflow in WSL2
	@echo   make airflow-deploy      - Deploy DAGs to WSL2 Airflow

# Install project dependencies and set up environment
install:
	@echo Installing project dependencies and setting up environment...
	@echo Creating virtual environment...
	$(PYTHON) -m venv .venv
	@echo Activating virtual environment and installing dependencies...
	.venv\Scripts\activate && python.exe -m pip install --upgrade pip
	.venv\Scripts\activate && pip install -r requirements.txt
	@echo Installation completed successfully!
	@echo To activate the virtual environment, run: .venv\Scripts\activate

# Clean up
clean:
	@echo Cleaning up artifacts...
	@if exist artifacts\models rmdir /s /q artifacts\models
	@if exist artifacts\data rmdir /s /q artifacts\data
	@if exist artifacts\evaluation rmdir /s /q artifacts\evaluation
	@if exist artifacts\predictions rmdir /s /q artifacts\predictions
	@if exist data\processed rmdir /s /q data\processed
	@if exist artifacts\encode rmdir /s /q artifacts\encode
	@if exist mlruns rmdir /s /q mlruns
	@mkdir artifacts\models 2>nul || echo.
	@mkdir artifacts\evaluation 2>nul || echo.
	@mkdir artifacts\predictions 2>nul || echo.
	@mkdir data\processed 2>nul || echo.
	@mkdir artifacts\encode 2>nul || echo.
	@echo Cleanup completed!

# Run data pipeline
data-pipeline:
	@echo Running data pipeline...
	@echo Setting PYTHONPATH to include src and utils directories...
	.venv\Scripts\activate && set PYTHONPATH=$(PYTHONPATH) && $(PYTHON) pipelines/data_pipeline.py

# Run training pipeline
train-pipeline:
	@echo Running training pipeline...
	@echo Setting PYTHONPATH to include src and utils directories...
	.venv\Scripts\activate && set PYTHONPATH=$(PYTHONPATH) && $(PYTHON) pipelines/training_pipeline.py

# Run streaming inference pipeline with sample JSON
streaming-inference:
	@echo Running streaming inference pipeline with sample JSON...
	@echo Setting PYTHONPATH to include src and utils directories...
	.venv\Scripts\activate && set PYTHONPATH=$(PYTHONPATH) && $(PYTHON) pipelines/streaming_inference_pipeline.py

# Run all pipelines in sequence
run-all:
	@echo Running all pipelines in sequence...
	@echo ========================================
	@echo Step 1: Running data pipeline
	@echo ========================================
	.venv\Scripts\activate && set PYTHONPATH=$(PYTHONPATH) && $(PYTHON) pipelines/data_pipeline.py
	@echo.
	@echo ========================================
	@echo Step 2: Running training pipeline
	@echo ========================================
	.venv\Scripts\activate && set PYTHONPATH=$(PYTHONPATH) && $(PYTHON) pipelines/training_pipeline.py
	@echo.
	@echo ========================================
	@echo Step 3: Running streaming inference pipeline
	@echo ========================================
	.venv\Scripts\activate && set PYTHONPATH=$(PYTHONPATH) && $(PYTHON) pipelines/streaming_inference_pipeline.py
	@echo.
	@echo ========================================
	@echo All pipelines completed successfully!
	@echo ========================================

# Launch MLflow UI
mlflow-ui:
	@echo Launching MLflow UI...
	@echo MLflow UI will be available at: http://localhost:$(MLFLOW_PORT)
	@echo Press Ctrl+C to stop the server
	.venv\Scripts\activate && mlflow ui --host 0.0.0.0 --port $(MLFLOW_PORT) --workers $(WORKERS)

# Stop all MLflow servers
stop-all:
	@echo Stopping all MLflow servers on port $(MLFLOW_PORT)...
	@for /f "tokens=5" %%a in ('netstat -ano ^| findstr :$(MLFLOW_PORT)') do taskkill /PID %%a /F >nul 2>&1
	@echo All MLflow servers on port $(MLFLOW_PORT) have been stopped!

# Sync DAGs from Windows to WSL2
sync-dags-to-wsl:
	@echo Syncing DAGs from Windows to WSL2...
	@wsl -d Ubuntu mkdir -p /home/nithira17/airflow-class/.airflow/dags/
	@if exist dags for %%f in (dags\*.py) do wsl -d Ubuntu cp "/mnt/c/Users/hewaj/Desktop/Zuu Crew/Customer Churn Prediction - AirFlow/dags/%%~nxf" "/home/nithira17/airflow-class/.airflow/dags/"
	@echo DAGs synced successfully!
	@echo Access Airflow UI at: http://localhost:8080

# Start Airflow in WSL2 (opens new terminal)
airflow-start-wsl:
	@echo Starting Airflow in WSL2...
	@echo This will open a new WSL2 terminal window
	wsl -d Ubuntu -e bash -c "cd ~/airflow-class && source .venv/bin/activate && ./start_airflow.sh"

# Check if Airflow is running
airflow-status:
	@echo Checking Airflow status...
	@curl -s http://localhost:8080/health || echo Airflow is not running
	@echo.
	@echo If running, access at: http://localhost:8080

# Stop Airflow (kills WSL2 processes)
airflow-stop-wsl:
	@echo Stopping Airflow in WSL2...
	@wsl -d Ubuntu pkill -f airflow || echo No Airflow processes found
	@echo Airflow stopped.

# Complete Airflow workflow
airflow-deploy:
	@echo Deploying to Airflow...
	@$(MAKE) sync-dags-to-wsl
	@echo DAGs deployed! Start Airflow with: make airflow-start-wsl
	@echo Or manually run in WSL2: cd ~/airflow-class && ./start_airflow.sh