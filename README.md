# TU-elevate
# Uber Smart Earner Assistant

An AI-powered assistant that helps Uber drivers maximize earnings through intelligent surge prediction and personalized recommendations.

## Project Overview

Our project addresses the challenge of helping Uber drivers optimize their earnings by providing:
- Real-time surge multiplier predictions using machine learning
- AI-powered recommendations for when and where to drive
- Driver wellness monitoring and break suggestions
- Data-driven insights based on weather, time, and location patterns

## Features

### Surge Prediction Model
We built a Random Forest regression model trained on historical data that predicts surge multipliers based on:
- Hour of the day
- City location
- Weather conditions
- Cancellation rates
- Predicted earnings per hour

### AI Recommendations
Our system provides context-aware advice using Hugging Face's language models. The recommendations are personalized based on:
- How long you've been driving
- Current weather conditions
- Predicted demand patterns
- Your location

### Interactive Web Interface
We designed a clean, mobile-optimized dashboard using Streamlit that mimics Uber's actual Trip Radar interface, showing:
- Real-time surge allocation
- Earnings projections
- Trip details and locations

## Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager
- Git

### Step-by-Step Setup

1. **Clone the repository**

git clone <your-repo-url>
cd uber-smart-earner-assistant



### Install dependencies

pip install -r requirements.txt

Set up your data files
Place your CSV data files in the data/ directory:


uber_hackathon_v2_mock_data.xlsx - surge_by_hour.csv
uber_hackathon_v2_mock_data.xlsx - weather_daily.csv
uber_hackathon_v2_mock_data.xlsx - cancellation_rates.csv
uber_hackathon_v2_mock_data.xlsx - heatmap.csv


### Train the model

cd notebooks
jupyter notebook model_training.ipynb
Run all cells in the notebook to train and save the model to models/surge_predictor_rf.pkl

### Create a .env file in the project root
echo "HUGGINGFACEHUB_API_TOKEN=your_token_here" > .env
Get a free token at https://huggingface.co/settings/tokens

### Start the Streamlit web interface:
streamlit run app_web.py
The app will automatically open in your browser at http://localhost:8501. You can adjust settings in the sidebar to see how different conditions affect surge predictions and recommendations.

### How It Works
Model Training
We trained a Random Forest model with 200 estimators on historical Uber data. The process involves:

Loading data from multiple sources (surge patterns, weather, cancellations, demand heatmaps)
Aggregating heatmap data to get city-level insights
Encoding categorical variables like weather conditions
Splitting data into training and test sets (80/20 split)
Training the model and evaluating performance
Saving the model for use in the app

The model achieved good performance on predicting surge multipliers based on the input features.
AI Integration
We integrated Hugging Face's Mistral-7B-Instruct model to generate natural language recommendations. The AI considers:

Current surge conditions from our ML model
A 24-hour forecast of surge patterns
How many hours the driver has worked
Weather conditions and their impact
City-specific demand patterns

### Technical Stack

Language: Python 3.13
Machine Learning: scikit-learn (Random Forest Regressor)
Web Framework: Streamlit
AI Integration: Hugging Face Inference API (Mistral-7B-Instruct)
Data Processing: pandas, numpy
Visualization: matplotlib
Model Persistence: joblib
