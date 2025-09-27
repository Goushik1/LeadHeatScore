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

# Node.js

# React 19+

# Tailwind CSS

# Chart.js

# Axios

# PapaParse

# Install frontend dependencies and start

cd frontend
npm install
npm start

## Data Setup

# Place CSV files inside the /data directory

# Configure API targets in a .env file

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
