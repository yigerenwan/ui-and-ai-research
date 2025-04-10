- Install the required packages:

```bash
pip install streamlit
```

- install lamma-cpp-python

for cpu
```bash
pip install lamma-cpp-python
```

for gpu
```bash
CMAKE_ARGS="-DGGML_CUDA=ON" pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python
```


- install huggingface-hub

```bash
pip install huggingface-hub
```

- select the model from huggingface

```
https://huggingface.co/bartowski/agentica-org_DeepCoder-14B-Preview-GGUF
```

- run the streamlit app

```bash
streamlit run streamlit.py
```
