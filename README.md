# Linear Algebra with CUDA

[![Continuous Integration Status](https://github.com/GDeLaurentis/linac-dev/actions/workflows/continuous_integration.yml/badge.svg)](https://github.com/GDeLaurentis/linac-dev/actions)
[![Coverage](https://img.shields.io/badge/Coverage-84%25-greenyellow?labelColor=2a2f35)](https://github.com/GDeLaurentis/linac-dev/actions)

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

Timing for Gaussian elimination to row reduced echelon form, with 32-bit integers (finite field) or single-precision floating-point numbers.

On an RTX 2080 Ti (on Merlin cluster)
- 8192 (8s) 
- 16384 (51s)
- 32768 (391s) 
- 40000 (793s)