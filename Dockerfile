FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir finpy_tse pandas

COPY test_tsetmc.py .

CMD ["python", "test_tsetmc.py"]
