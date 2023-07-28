# Tomography Alignment
Rigid body alignment for x-ray tomography data

# Installation

The library dependencies and packaging is managed with [Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer). After installation of poetry, please run in the repository's root folder:

```bash
poetry install
```

This will install the `tomoalign` package from [pyproject.toml](./pyproject.toml) in a separate virtual environment. To activate the virtual environment simply type `poetry shell` (at repository's root folder) and to deactive the environment `deactivate`.

## Fortran code compilation

After installation of the dependencies, this step wraps the relevant fortran code by using the `numpy.f2py` utility. This is done by compiling the sources and building an extension module containing the wrappers.

Assuming you are at the repository's root folder execute this bash file (ignore the warning messages):

```bash
bash f2py.sh
```

