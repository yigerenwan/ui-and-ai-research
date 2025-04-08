## 🔨 Installation

Clone the repo:
```bash
git clone https://github.com/VAST-AI-Research/TripoSG.git
cd TripoSG
```

Create a conda environment (optional):
```bash
conda create -n tripoSG python=3.10
conda activate tripoSG
conda install -c conda-forge gcc gxx
conda install -c conda-forge libstdcxx-ng
```

Install dependencies:
```bash
# pytorch (select correct CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/{your-cuda-version}

# other dependencies
pip install -r requirements.txt
```

for cuda 12.4
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

## 💡 Quick Start

Generate a 3D mesh from an image:
```bash
python -m scripts.inference_triposg --image-input assets/example_data/hjswed.png
```

The required model weights will be automatically downloaded:
- TripoSG model from [VAST-AI/TripoSG](https://huggingface.co/VAST-AI/TripoSG) → `pretrained_weights/TripoSG`
- RMBG model from [briaai/RMBG-1.4](https://huggingface.co/briaai/RMBG-1.4) → `pretrained_weights/RMBG-1.4`

## 💻 System Requirements

- CUDA-enabled GPU with at least 8GB VRAM

## 📝 Tips

- If you want to use the full VAE module (including the encoder part), you need to uncomment the Line-15 in `triposg/models/autoencoders/autoencoder_kl_triposg.py` and install `torch-cluster`. and run:
```
python -m scripts.inference_vae --surface-input assets/example_data_point/surface_point_demo.npy
```

Install streamlit
```bash
pip install streamlit
```

Run streamlit
```bash
streamlit run streamlit.py
```