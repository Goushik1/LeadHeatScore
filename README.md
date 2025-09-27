# Project Dashboard Setup

## Backend Requirements (requirements.txt)

flask
flask-cors
pandas
scikit-learn
xgboost
matplotlib
langchain
langchain-groq
sentence-transformers
tqdm
requests
python-dotenv
pymongo
elasticsearch
numpy

# Python ≥ 3.10 required

# Install backend dependencies

pip install -r requirements.txt

## Frontend Requirements

Node.js
React 19+
Tailwind CSS
Chart.js
Axios
PapaParse

# Install frontend dependencies and start

cd frontend
npm install
npm start

## Data Setup

# Place CSV files inside the /data directory

# Configuration

Create a .env file in the root directory and add your credentials:

# MongoDB

MONGO_URI="your_mongodb_connection_string"

# Elasticsearch

ELASTIC_USERNAME="your_elasticsearch_username"
ELASTIC_PASSWORD="your_elasticsearch_password"  
ELASTIC_HOST="your_elasticsearch_host"

# API Key

API_KEY="your_api_key_here"

## Run Instructions

# 1. Start the Flask backend

cd backend
python app.py

# 2. Launch the React frontend (in separate terminal)

cd frontend
npm start

# 3. Open the dashboard in your browser at:

# http://localhost:3000

## Available API Endpoints:

# /score

# /metrics

# /abtestscore

# /abtestsummary
