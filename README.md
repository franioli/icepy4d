# Welcome to ICEpy4D

4D Image-based Continuos monitoring of glaciers' Evolution with low-cost stereo-cameras and Deep Learning photogrammetry.

`ICEpy4D` is a under active development.

## Installation guide

##### Requirements

- 64-bit Python `>= 3.8` but `< 3.10`
- a NVIDIA graphic card with CUDA capability is strongly reccomended.

##### Create Anaconda environment

```bash
conda create -n icepy4d python=3.8
conda activate icepy4d
```

##### Install gdal for raster manipulation

First, install `gdal`, which is required for building and manipulating orthophotos and DSMs. As the dependacies of `gdal`, are quite strict, it is suggested to install it first with conda. If you don't intend to build orthophotos and DSM, you can skip this step (be careful to remove rasterio from requirements.txt as well in order to avoid unsatisfied dependencies).

```bash
conda update -n base -c conda-forge conda
conda install -c conda-forge gdal
```

Check that `gdal` is correctly installed with:

```bash
python -c "from osgeo import gdal"
```

##### Install Pytorch

Install pythorch following the official guidelines (<https://pytorch.org/get-started/locally/>). Be careful to select the correct CUDA version as that installed on your system.

```bash
pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu116

```

##### Install other required packages

```bash
pip3 install -r requirements.txt
```

##### Various

- If you intend to use Jupyter Notebooks for running main scripts, install jupyterlab with

```bash
pip3 install jupyterlab
```

- When using VScode and Matplotlib, use TkAgg as interactive backend.

##### Install ICEpy4D

Install ICEpy4D package by running from the root folder

```bash
pip install -e .
```

and try to import it

```bash
conda activate icepy4d
python -c "import icepy4d"
```

If no error is given, ICEpy4D is successfully installed and it can be imported with `import icepy4d`


#### For contributing
`bash
pip install -r requirements-dev.txt
pre-commit install
`
