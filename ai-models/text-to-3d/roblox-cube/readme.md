

- Clone and install this repo in a virtual environment, via:

```bash
git clone https://github.com/Roblox/cube.git
cd cube
pip install -e .[meshlab]
```

- Download the model weights from hugging face or use the huggingface-cli:

```bash
huggingface-cli download Roblox/cube3d-v0.1 --local-dir ./model_weights
```

- Run the streamlit app:
```bash
streamlit run streamlit.py
```
 You need to install these packages:
 ```bash
 pip install streamlit trimesh matplotlib
 ```
