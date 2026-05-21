<p align="center">
  <img src="https://raw.githubusercontent.com/GDeLaurentis/linac/main/assets/linac_logo_v2_transparent.png"
       alt="Linac: Linear algebra with CUDA"
       width="550">
</p>

[![CI Lint](https://github.com/GDeLaurentis/linac-dev/actions/workflows/ci_lint.yml/badge.svg)](https://github.com/GDeLaurentis/linac-dev/actions/workflows/ci_lint.yml)
[![CI Test](https://github.com/GDeLaurentis/linac-dev/actions/workflows/ci_test.yml/badge.svg)](https://github.com/GDeLaurentis/linac-dev/actions/workflows/ci_test.yml)
[![Coverage](https://img.shields.io/badge/Coverage-81%25-greenyellow?labelColor=2a2f35)](https://github.com/GDeLaurentis/linac-dev/actions)
[![Docs](https://github.com/GDeLaurentis/linac/actions/workflows/cd_docs.yml/badge.svg?label=Docs)](https://gdelaurentis.github.io/linac/)
[![PyPI](https://img.shields.io/pypi/v/linac?label=PyPI)](https://pypi.org/project/linac/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/linac.svg?label=PyPI%20downloads)](https://pypi.org/project/linac/)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/GDeLaurentis/linac/HEAD)
[![DOI](https://zenodo.org/badge/xxxxxxxx.svg)](https://zenodo.org/doi/10.5281/zenodo.20327732)
[![Python](https://img.shields.io/pypi/pyversions/linac?label=Python)](https://pypi.org/project/linac/)

`Linac` implements GPU-accelerated linear algebra using CUDA, with support for real and complex floating-point arithmetic, finite fields, and leading-digit p-adic numbers. Its main functionality is dense Gaussian elimination to reduced row-echelon form. Applications include functional reconstruction of analytic scattering amplitudes.

## Installation

The latest stable release is available from [PyPI](https://pypi.org/project/linac/):

```bash
pip install linac
```

To enable GPU acceleration, install the CUDA extra:

```bash
pip install "linac[cuda]"
```

For development, clone the repository and install it in editable mode:

```bash
git clone https://github.com/GDeLaurentis/linac.git
pip install -e "linac[full]"
```

Available extras are:

```text
cuda, dev, full(=cuda+dev)
```

## Requirements

The core package depends on:

```text
numpy, pycoretools, pyadic, syngular
```

Optional dependencies include:

```text
pycuda   # CUDA/GPU support
```

with the cuda extra, and 

```text
mpmath, galois, diskcache, pytest, pytest-cov, flake8
```

with the dev extra.

GPU acceleration requires a working CUDA development environment, including `nvcc`, compatible NVIDIA drivers, and access to a CUDA-enabled GPU.

You can check your CUDA setup with:

```bash
nvcc --version
nvidia-smi
```

## Quick Start

The main entry point is `cuda_row_reduce`, which computes a row-reduced echelon form on the GPU.

```python
import numpy
from linac import cuda_row_reduce

p = 2**31 - 1

A = numpy.random.randint(0, p, size=(2000, 2000), dtype="uint32")
rref = cuda_row_reduce(A, field_characteristic=p)
```

Setting `field_characteristic=0` selects floating-point arithmetic:

```python
A = numpy.random.rand(2000, 2000)
rref = cuda_row_reduce(A, field_characteristic=0)
```

For CPU row reduction, use:

```python
from linac import row_reduce

rref, variable_order = row_reduce(A, prime=p)
```

`Linac` also provides utilities for constructing dense linear systems from polynomial ansätze:

```python
from linac import load_matrices

matrices = load_matrices(prefactors, ansatze, points, use_cuda=True)
```

## Testing

Run the test suite with:

```bash
pytest -rs --verbose --cov=linac --cov-report=html
```

## Timings

The following timings are for dense Gaussian elimination to reduced row-echelon form over the finite field with `p=(2**31 - 1)`. Times are shown in seconds as mean ± standard deviation.

| Matrix size | A100 80GB | RTX 4070 Laptop 8GB |
|:-----------:|----------:|--------------------:|
| 1,000       | 0.52 ± 0.01 | 0.28 ± 0.02        |
| 2,000       | 0.56 ± 0.01 | 0.34 ± 0.01        |
| 5,000       | 1.02 ± 0.01 | 3.31 ± 0.01        |
| 10,000      | 3.60 ± 0.01 | 25.97 ± 0.66       |
| 15,000      | 10.23 ± 0.02 | 98.21 ± 0.85      |
| 20,000      | 23.40 ± 0.03 | 241.29 ± 3.00     |
| 100,000     | 2824.64 | —                  |

The 100k run used approximately 37.67 GiB of memory on an 80 GiB A100.

For full benchmark details and comparison with CPU implementations, see the accompanying paper and documentation.

## Size Limit

For 32-bit entries, the approximate upper bound for a dense square matrix is

```text
N_max ~ sqrt(VRAM [bytes] / 4)
```

| Available VRAM | Approx. square matrix size |
|:--------------:|---------------------------:|
| 4 GB           | 31,622                     |
| 8 GB           | 44,721                     |
| 11 GB          | 52,440                     |
| 12 GB          | 54,772                     |
| 16 GB          | 63,245                     |
| 24 GB          | 77,459                     |
| 40 GB          | 100,000                    |
| 80 GB          | 141,421                    |

The practical limit is usually close to this estimate, up to padding and (small) memory-management overheads.

## Documentation

Documentation is available at:

```text
https://gdelaurentis.github.io/linac/
```

## Citation

If you use `Linac` in academic work, please cite the accompanying paper and the Zenodo archive:

```bibtex
@software{linac,
  author = {De Laurentis, Giuseppe and Franklin, Jack},
  title = {Linac: linear algebra with CUDA over finite fields},
  year = {2026},
  doi = {10.5281/zenodo.xxxxxxxx},
}
```