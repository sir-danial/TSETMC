FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# نصب وابستگی‌ها
RUN pip install --no-cache-dir \
    django \
    gunicorn \
    finpy_tse \
    pandas \
    numpy \
    ta

# کپی پروژه
COPY . .

# جمع‌آوری فایل‌های استاتیک (الان چیزی نداره ولی استاندارده)
RUN python manage.py collectstatic --noinput || true

# پورت مورد استفاده دارکوب
EXPOSE 8000

# اجرای سرور production
CMD ["gunicorn", "config.wsgi:application", "--bind", "0.0.0.0:8000"]
