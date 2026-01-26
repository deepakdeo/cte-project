FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

WORKDIR /app

# System deps for native extensions (e.g., wordcloud)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        libfreetype6-dev \
        libpng-dev \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# System deps for poetry
RUN pip install --no-cache-dir poetry

COPY pyproject.toml poetry.lock README.md /app/
RUN poetry install --no-ansi --no-root

COPY src /app/src
COPY scripts /app/scripts
RUN poetry install --no-ansi --only-root

EXPOSE 8501

CMD ["poetry", "run", "streamlit", "run", "scripts/cte_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
