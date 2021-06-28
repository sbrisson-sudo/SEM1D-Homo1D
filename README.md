# SEM1D-Homo1D

### Main Programs:
- `SEM1D.py` : python solver program of the shear elastic wave equation in 1D (Spectral Element Method)
- `Homo1D.py` : python homogeneisation program of the medium, density and shear modulus (2-scale non-periodic 1D homogeneisation)
> For their use, refer to the `main.py` program.

### Tools Programs:
- `plot1Dnc.py` : tool to plot the nc files produced by `SEM1D.py`

### Data directories
- `data` : netCDF files produces by `SEM1D.py`
- `xwh_gll` : data needed for the SEM

### Usage Example
  ```bash
  $ export PYTHONPATH=$(pwd)
  $ python main.py
  $ ./plot1Dnc.py data/ref.nc data/homo*
  ```
