# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html),
with the exception that new major versions may also accompany paper releases.


## [Unreleased]

### Added

### Changed

### Fixed

### Deprecated


## [1.0.1] - 2026-05-21

### Fixed

- Switched on DOI generation with Zenodo
- PyPI needs absolute path for logo


## [1.0.0] - 2026-05-21

### Added

- GPU row reduction via `cuda_row_reduce`.
- CPU row reduction via `row_reduce`.
- GPU construction of dense linear systems via `cuda_load_matrix`.
- Higher-level matrix construction interface via `load_matrices`.
- Tensor-valued callable wrapper `tensor_function` for callables returning NumPy arrays.
- Vector spaces over number fields via `ColumnVectorSpace`.
- Vector spaces over function fields via `VectorSpaceOfFunctions`.
- Sparse-matrix utilities, including coordinate-format conversions such as `coo_from_matrix` and `matrix_from_coo`.
- Linear-algebra utilities, including `pivot_columns_from_row_reduced_echelon_form` and `canonical_kernel_from_row_reduced_echelon_form`.


[unreleased]: https://github.com/GDeLaurentis/linac/compare/v1.0.1...HEAD
[1.0.1]: https://github.com/GDeLaurentis/lips/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/GDeLaurentis/linac/releases/tag/v1.0.0