FROM python:3.11-slim

WORKDIR /app

# Set UTF-8 locale (fixes umlaut/special character handling)
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONIOENCODING=utf-8

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download en_core_web_sm && \
    python -m spacy download de_core_news_sm

COPY app.py .
COPY glossaries/ glossaries/

EXPOSE 7860

CMD ["python", "app.py"]
