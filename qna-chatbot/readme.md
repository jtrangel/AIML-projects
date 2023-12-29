Project basis documentation: https://towardsdatascience.com/build-a-q-a-app-with-pytorch-cb599480e29

### Directory includes:
- `app/main.py` script - FastAPI app for setting up API endpoints
- `app/classes.py` script - Define classes for taking context, processing input data into vector
   embeddings, and resolving an answer for a query.
- `app/test.py` - for testing local functionality of classes/chatbot
- `app/test_container.py` - for testing containerized API functionality of chatbot
- `Dockerfile` - for building Docker image
- `train-v2.0.json` - Stanford Question Answering Dataset 2.0
- `download_model.sh` - huggingface model downloader shell script

### Notes:

1. Use `sh download_model.sh` instead of `bash download_model.sh`
2. Make sure to have `wget.exe` in your bash binaries (`./usr/bin`) directory. Despite similarity to `curl`, functionality differs with parameters
3. For any python packages, add `python -m` for pip
    - i.e. `python -m pip install transformers[torch]`
4. Rust and Visual Studio is a prerequisite for transformers. Install via `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`. I believe this is for metadata (`*.toml`) files
5. Encountered a subprocess issue with building wheel for sentencepiece. Solution was to create a conda venv with python
version of 3.8.12. 
```bash
conda create -n py38 python=3.8.12

source activate py38
```

### Running app via docker and fastapi def
```commandline
docker build . -t qamodel &&\
  docker run -p 8000:8000 qamodel
```