set USE_FLASH_ATTENTION=1
pip uninstall torch
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu124
pip install build
pip install cmake
pip install ninja
pip install wheel
pip install flash-attn --no-build-isolation 
