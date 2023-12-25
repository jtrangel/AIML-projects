Project basis documentation: https://towardsdatascience.com/build-a-q-a-app-with-pytorch-cb599480e29

### Directory includes:
- main.py script
- training data json
- huggingface model downloader shell script

### Notable changes:

1. Use `sh download_model.sh` instead of `bash download_model.sh`
2. Make sure to have `wget.exe` in your bash binaries (`./usr/bin`) directory. Despite similarity to `curl`, functionality differs with parameters
3. For any python packages, add `python -m` for pip
    - i.e. `python -m pip install transformers[torch]`
4. Rust and Visual Studio is a prerequisite for transformers. Install via `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`. I believe this is for metadata (`*.toml`) files
5. Encountered a subprocess issue with building wheel for sentencepiece