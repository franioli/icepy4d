# belpy

This is a work-in-progress repository.

## Installation guide

conda create -n belpy python=3.8
conda activate belpy
conda update -n base -c conda-forge conda
conda install -c conda-forge gdal
python -c "from osgeo import gdal"
pip3 install torch torchvision --extra-index-url <https://download.pytorch.org/whl/cu116> (Depending on CUDA VERSION, see <https://pytorch.org/get-started/locally/>)
pip3 install numpy opencv-python tqdm matplotlib plotly scipy h5py pycolmap open3d kornia gdown rasterio exifread easydict
pip3 install pydegensac