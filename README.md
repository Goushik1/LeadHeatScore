# Lead Classification & Outreach Platform

This project is a full-stack application for lead scoring, outreach recommendation, and performance metrics visualization. It combines a Python (Flask) backend and React.js dashboard for interactive data handling.

# 1. Create and activate a Python virtual environment (backend)

```
cd backend
python -m venv venv
venv\Scripts\activate
```

# 2. Install backend requirements

```
pip install -r requirements.txt
```

# 3. Add your environment variables to a .env file

Create a file named `.env` in the root directory with the following content:

```
MongoDB
MONGO_URI="your_mongodb_connection_string"

Elasticsearch
ELASTIC_USERNAME="your_elasticsearch_username"
ELASTIC_PASSWORD="your_elasticsearch_password"
ELASTIC_HOST="your_elasticsearch_host"

API Key
API_KEY="your_api_key_here"
```

# 4. Start the Flask backend

```
python app.py
```

# 5. Set up and run the frontend (React dashboard)

```
cd frontend
npm install
npm start
```

# 6. Access the app

# - Flask backend API at: http://localhost:5000

# - React dashboard at: http://localhost:3000

---

## API Usage

| Endpoint       | Method | Description       |
| -------------- | ------ | ----------------- |
| /score         | POST   | Score single lead |
| /scorecsv      | POST   | Batch score CSV   |
| /metrics       | GET    | Model metrics     |
| /recommend     | POST   | Outreach message  |
| /calibration   | POST   | Calibration plot  |
| /abtestscore   | POST   | AB score submit   |
| /abtestsummary | GET    | AB score summary  |

---
