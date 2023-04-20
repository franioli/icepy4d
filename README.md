# Welcome to ICEpy4D

4D Image-based Continuos monitoring of glaciers' Evolution with low-cost stereo-cameras and Deep Learning photogrammetry.

`ICEpy4D` is a under active development.

## Installation guide

### Requirements

- 64-bit Python `>= 3.8` but `< 3.10`
- a NVIDIA graphic card with CUDA capability is strongly reccomended.

### Create Anaconda environment

```bash
conda create -n icepy4d python=3.8
conda activate icepy4d
```

Install Icepy4D from PyPi repository
```bash
pip install icepy4d
```

Install Metashape Python API (they can be downloaded from [https://www.agisoft.com/downloads/installer/](https://www.agisoft.com/downloads/installer/))
```bash
wget https://s3-eu-west-1.amazonaws.com/download.agisoft.com/Metashape-1.8.5-cp35.cp36.cp37.cp38-abi3-linux_x86_64.whl
pip install Metashape-1.8.5-cp35.cp36.cp37.cp38-abi3-linux_x86_64.whl
```

Try to import ICEpy4D package
```bash
conda activate icepy4d
python -c "import icepy4d"
```

If no error is given, ICEpy4D is successfully installed and it can be imported within your script with `import icepy4d`


#### For contributing
Install additional requirements for development:

`bash
pip install -r requirements-dev.txt
pre-commit install
`
