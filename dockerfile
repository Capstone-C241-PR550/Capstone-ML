FROM python:3.8


COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
# Expose the port for Flask app
EXPOSE 5000
# Define environment variable
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

CMD ["flask", "run"]