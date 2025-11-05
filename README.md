# Linear Algebra with CUDA

[![CI Lint](https://github.com/GDeLaurentis/linac-dev/actions/workflows/ci_lint.yml/badge.svg)](https://github.com/GDeLaurentis/linac-dev/actions/workflows/ci_lint.yml)
[![CI Test](https://github.com/GDeLaurentis/linac-dev/actions/workflows/ci_test.yml/badge.svg)](https://github.com/GDeLaurentis/linac-dev/actions/workflows/ci_test.yml)
[![Coverage](https://img.shields.io/badge/Coverage-81%25-greenyellow?labelColor=2a2f35)](https://github.com/GDeLaurentis/linac-dev/actions)
[![Docs](https://github.com/GDeLaurentis/linac/actions/workflows/cd_docs.yml/badge.svg?label=Docs)](https://gdelaurentis.github.io/linac/)
[![PyPI](https://img.shields.io/pypi/v/linac?label=PyPI)](https://pypi.org/project/linac/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/linac.svg?label=PyPI%20downloads)](https://pypi.org/project/linac/)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/GDeLaurentis/linac/HEAD)
[![DOI](https://zenodo.org/badge/xxxxxxxx.svg)](https://zenodo.org/doi/10.5281/zenodo.xxxxxxxx)
[![Python](https://img.shields.io/pypi/pyversions/linac?label=Python)](https://pypi.org/project/linac/)

The `Linac` library implements hardware accelaration for linear algebra using CUDA, over a variety of number fields, including finite fields.

## Installation
The package is available on the [Python Package Index](https://pypi.org/project/linac/)
```
pip install linac
```
Alternativelty, it can be installed by cloning the repo
```
git clone https://github.com/GDeLaurentis/linac.git path/to/repo
pip install -e path/to/repo[extras]
```
where `extras` can be any of
```
cuda, dev, full
```

## Requirements
`pip` will automatically install the required packages, which are
```
numpy, mpmath, pycuda (optional)
```
The GPU capabilities require a working CUDA environment.

## Testing

```
pytest3 -rs --verbose --cov=linac --cov-report=html
```

## Timings

Timing for Gaussian elimination to row reduced echelon form, with 32-bit integers (finite field). 
Single-precision floating-point numbers should have similar timings.

On an RTX 2080 Ti 11Gb (on Merlin cluster)

| Matrix size | Timing (seconds) |
|:-----------:|:----------------:|
|    8192     |        8         |
|    16384    |        51        |
|    32768    |       391        |
|    40000    |       793        |
|    50000    |      1899        |

## Theoretical Size Limit

The theoretical size limit is given by $\sqrt{\text{VRAM in GB}/ 4 * 10 ^ 9}$.
The real-world size limit is fairly close.

| Available VRAM | Square matrix size |
|:--------------:|:------------------:|
|       4        |       31622        |
|       8        |       44721        |
|       11       |       52440        |
|       12       |       54772        |
|       16       |       63245        |
|       24       |       77459        |
|       40       |       100000       |