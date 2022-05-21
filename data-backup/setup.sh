sudo docker run -d --gpus all -it --rm -v`pwd`/data:/data/ nvcr.io/nvidia/pytorch:21.06-py3
sudo docker run -d --gpus all -it --rm -v`pwd`/data:/data/ jp-tts:latest
pip install -q espnet==0.10.6 pyopenjtalk==0.2 pypinyin==0.44.0 parallel_wavegan==0.5.4 espnet_model_zoo
pip install --upgrade --no-cache-dir gdown