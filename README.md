# Linear Algebra with CUDA

[![Continuous Integration Status](https://github.com/GDeLaurentis/linac-dev/actions/workflows/continuous_integration.yml/badge.svg)](https://github.com/GDeLaurentis/linac-dev/actions)
[![Coverage](https://img.shields.io/badge/Coverage-0%25-red?labelColor=2a2f35)](https://github.com/GDeLaurentis/linac-dev/actions)

## Requirements
```
numpy, mpmath, pycuda (optional)
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