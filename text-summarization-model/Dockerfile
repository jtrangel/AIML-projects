FROM python:3.11-slim

# Specify working directory
WORKDIR /app

# Install required pip libraries
COPY requirements.txt .
# Optional default timeout setting for slower internet connection speeds
RUN pip install --default-timeout=100 --no-cache-dir -r ./requirements.txt

# copies the app files to the docker image
COPY /app /app

# runs our application at the start of the docker image
CMD ["streamlit", "run", "app.py", "--server.port=8501"]