# End-to-End ML Deployment

This repository demonstrates an end-to-end machine learning project for predicting student math scores based on demographic and academic features. The project covers the full ML lifecycle, from data ingestion and preprocessing to model training, evaluation, and deployment as a web application. It also includes Docker support for cloud deployment (AWS, Azure).

## Features
- Data ingestion, cleaning, and splitting
- Data transformation (imputation, encoding, scaling)
- Model training and hyperparameter tuning (multiple regressors)
- Model evaluation and selection
- Web app (Flask) for interactive predictions
- Dockerized for easy deployment

## Project Structure
- `notebook/eda.ipynb`, `notebook/train.ipynb`: EDA and model training experiments
- `src/components/`: Core pipeline modules (ingestion, transformation, training)
- `src/pipeline/`: Prediction pipeline for web app
- `app.py`: Flask web server for prediction UI
- `artifacts/`: Stores trained models and preprocessors
- `requirements.txt`: Python dependencies
- `Dockerfile`: Containerization for deployment

## Usage
### 1. Local Setup
Install dependencies:
```bash
pip install -r requirements.txt
```
Run the web app:
```bash
python app.py
```
Visit `http://localhost:5000` in your browser to use the prediction UI.

### 2. Docker Deployment
Build and push the Docker image:
```bash
docker build -t <your-repo>/mltest:latest .
docker login <your-repo>
docker push <your-repo>/mltest:latest
```

## How it Works
1. **Data Ingestion:** Reads student data, splits into train/test sets.
2. **Data Transformation:** Handles missing values, encodes categoricals, scales features.
3. **Model Training:** Trains and tunes several regressors, selects the best model.
4. **Prediction Pipeline:** Loads the trained model and preprocessor for inference.
5. **Web App:** Users input student features and receive predicted math scores.

## Example Input Features
- Gender
- Race/Ethnicity
- Parental Level of Education
- Lunch Type
- Test Preparation Course
- Reading Score
- Writing Score