# Installation guide

### Requirements

- 64-bit Python `= 3.8`
- a NVIDIA graphic card with CUDA capability is strongly reccomended.

`Python 3.8` is required to correctely install PyDegensac, that is used for geometric verification. Pydegensac installation fails with Python 3.9 and above. (see <https://github.com/ducha-aiki/pydegensac/issues/15#issue-1361049626>).

##### Create Anaconda environment

```bash
conda create -n icepy python=3.8
conda activate icepy
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

For installing required dependencies:
```bash
pip3 install -r requirements.txt
```

If you want to contribute to ICEpy4D, you need to install dev dependencies by running:
```bash
pip3 install -r requirements-dev.txt
```

##### Install ICEpy4D

Install ICEpy4D package by running from the root folder

```bash
pip install -e .
```

and try to import it

```bash
conda activate icepy
python -c "import icepy"
```

If no error is given, ICEpy4D is successfully installed and it can be imported with `import icepy`
