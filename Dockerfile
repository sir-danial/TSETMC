FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# نصب پکیج‌های سیستمی لازم
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# نصب وابستگی‌های پایتون
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# کپی کل پروژه
COPY . .

# collectstatic
RUN python manage.py collectstatic --noinput

# پورت
EXPOSE 8000

# اجرای پروژه
CMD ["gunicorn", "config.wsgi:application", "--bind", "0.0.0.0:8000"]
