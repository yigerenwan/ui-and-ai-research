pip install "flash-attn @ git+https://github.com/Dao-AILab/flash-attention.git#egg=flash-attn"

pip uninstall flash-attn
pip cache purge
pip install flash-attn --no-build-isolation

