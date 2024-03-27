FROM python:3.11.3-slim

WORKDIR /app

COPY test_reqs.txt /app/
RUN pip install -r test_reqs.txt

COPY . /app/

CMD python manage.py runserver 0.0.0.0:8000
