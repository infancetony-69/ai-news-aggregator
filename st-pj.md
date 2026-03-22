docker start ai-news-aggregator-db
.venv/Scripts/python main.py
.venv/Scripts/uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
