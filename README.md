# Welcome to ICEpy4D

4D Image-based Continuos monitoring of glaciers' Evolution with low-cost stereo-cameras and Deep Learning photogrammetry.

`ICEpy4D` is a under active development.
## Gettin Started

To get started with `ICEpy4D` you can refer documentation at [https://franioli.github.io/icepy4d/](https://franioli.github.io/icepy4d/) (please, note the documentation is currently under development).  
You can find some Jupyter Notebooks with examples in the `notebooks` folder.

## Requirements

- 64-bit Python `>= 3.8` but `< 3.10`
- a NVIDIA graphic card with CUDA capability is strongly reccomended.

## Installation guide

Create a new Anaconda environment

```bash
conda create -n icepy4d python=3.9
conda activate icepy4d
```

Install Icepy4D from PyPi repository

```bash
pip install icepy4d
```

or install it from source by cloning the repository and installing it with `pip`

```bash
git clone https://github.com/franioli/icepy4d.git
cd icepy4d
pip install -e .
```

In case of any error when installing `ICEpy4D` from PyPi, try to install it from source.

Install Metashape Python API for Bundle Adjustment and Dense reconstruction.
Metashape Python API can be downloaded from [https://www.agisoft.com/downloads/installer/](https://www.agisoft.com/downloads/installer/) or use `wget` (under Linux).

```bash
wget https://s3-eu-west-1.amazonaws.com/download.agisoft.com/Metashape-1.8.5-cp35.cp36.cp37.cp38-abi3-linux_x86_64.whl
pip install Metashape-1.8.5-cp35.cp36.cp37.cp38-abi3-linux_x86_64.whl
```

You need to have a valid Metashape license to use the API and you need to activate it (see [https://github.com/franioli/metashape](https://github.com/franioli/metashape) for how to do it)

Try to import ICEpy4D package

```bash
conda activate icepy4d
python -c "import icepy4d"
```

If no error is given, ICEpy4D is successfully installed and it can be imported within your script with `import icepy4d`

## Cite ICEpy4D
If you use `ICEpy4D` in your research, please cite it as:

@article{ioli2024,
  title={Deep Learning Low-cost Photogrammetry for 4D Short-term Glacier
Dynamics Monitoring},
  author={Ioli, Francesco and Dematteis, Nicolò and Giordan, Daniele and Nex, Francesco and Pinto Livio},
  journal={PFG – Journal of Photogrammetry, Remote Sensing and Geoinformation Science},
  year={2024},
  DOI = {10.1007/s41064-023-00272-w}
}


```bibtex
@article{ioli2023replicable,
  title={A Replicable Open-Source Multi-Camera System for Low-Cost 4d Glacier Monitoring},
  author={Ioli, F and Bruno, E and Calzolari, D and Galbiati, M and Mannocchi, A and Manzoni, P and Martini, M and Bianchi, A and Cina, A and De Michele, C and others},
  journal={The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences},
  doi={10.5194/isprs-archives-XLVIII-M-1-2023-137-2023},
  url={[https://doi.org/10.5194/isprs-archives-XLVIII-M-1-2023-137-2023](https://doi.org/10.5194/isprs-archives-XLVIII-M-1-2023-137-2023)},
  volume={XLVIII-M-1-2023},
  pages={137--144},
  year={2023},
  publisher={Copernicus GmbH}
}
```

### For contributing

Install additional requirements for development:

```bash
pip install -e .[dev]
pre-commit install
```

## Permissions and acknowledgements
