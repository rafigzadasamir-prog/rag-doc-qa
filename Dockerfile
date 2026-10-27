FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY data ./data

ENV PYTHONPATH=/app/src
ENV CHROMA_PERSIST_DIR=/app/chroma_data

EXPOSE 8000

CMD ["uvicorn", "rag_doc_qa.api:app", "--host", "0.0.0.0", "--port", "8000"]
