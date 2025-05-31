## Foward Warp Pytorch Version

`[25-05-31]` Updated to work on pytorch=2.5.1, python-3.12, CUDA=12.9

### Install

```bash
export CUDA_HOME=/usr/local/cuda #use your CUDA instead
chmod a+x install.sh
./install.sh
```

### Test

```bash
cd test
python test.py
```

### Usage

```python
from Forward_Warp import forward_warp

# default interpolation mode is Bilinear
fw = forward_warp()
im2_bilinear = fw(im0, flow) 
# use interpolation mode Nearest
# Notice: Nearest input-flow's gradient will be zero when at backward.
fw = forward_warp(interpolation_mode="Nearest")  
im2_nearest = fw(im0, flow) 
```
