FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps for poetry
RUN pip install --no-cache-dir poetry

COPY pyproject.toml poetry.lock README.md /app/
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

COPY . /app

EXPOSE 8501

CMD ["poetry", "run", "streamlit", "run", "scripts/cte_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
