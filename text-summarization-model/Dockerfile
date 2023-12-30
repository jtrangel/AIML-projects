FROM python:3.11.7

# Install required pip libraries
COPY requirements.txt ./requirements.txt
RUN apt-get update &&\
    rm -rf /var/lib/apt/lists/* &&\
    pip install --no-cache-dir -r ./requirements.txt

# copies the app files to the docker image
COPY app/ app/
WORKDIR app/

EXPOSE 8501

# runs our application at the start of the docker image
CMD streamlit run app.py --server.port 8501 --server.address 127.0.0.1