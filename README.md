# Linear Algebra with CUDA

[![CI Lint](https://github.com/GDeLaurentis/linac-dev/actions/workflows/ci_lint.yml/badge.svg)](https://github.com/GDeLaurentis/linac-dev/actions/workflows/ci_lint.yml)
[![CI Test](https://github.com/GDeLaurentis/linac-dev/actions/workflows/ci_test.yml/badge.svg)](https://github.com/GDeLaurentis/linac-dev/actions/workflows/ci_test.yml)
[![Coverage](https://img.shields.io/badge/Coverage-81%25-greenyellow?labelColor=2a2f35)](https://github.com/GDeLaurentis/linac-dev/actions)

## Requirements
```
numpy, mpmath, pycuda (optional), gmpTools (optional)
```

## Installation
```
pip install -e path/to/repo
```
or
```
pip install -e path/to/repo[full]
```

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