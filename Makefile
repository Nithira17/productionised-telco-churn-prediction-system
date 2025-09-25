.PHONY: all clean install train-pipeline data-pipeline streaming-inference run-all mlflow-ui help

# Default Python interpreter
PYTHON = python
VENV = .venv/Scripts/activate

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
	.venv\Scripts\activate && $(PYTHON) pipelines/data_pipeline.py

# Run training pipeline
train-pipeline:
	@echo Running training pipeline...
	.venv\Scripts\activate && $(PYTHON) pipelines/training_pipeline.py

# Run streaming inference pipeline with sample JSON
streaming-inference:
	@echo Running streaming inference pipeline with sample JSON...
	.venv\Scripts\activate && $(PYTHON) pipelines/streaming_inference_pipeline.py

# Run all pipelines in sequence
run-all:
	@echo Running all pipelines in sequence...
	@echo ========================================
	@echo Step 1: Running data pipeline
	@echo ========================================
	.venv\Scripts\activate && $(PYTHON) pipelines/data_pipeline.py
	@echo.
	@echo ========================================
	@echo Step 2: Running training pipeline
	@echo ========================================
	.venv\Scripts\activate && $(PYTHON) pipelines/training_pipeline.py
	@echo.
	@echo ========================================
	@echo Step 3: Running streaming inference pipeline
	@echo ========================================
	.venv\Scripts\activate && $(PYTHON) pipelines/streaming_inference_pipeline.py
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