import shutil


try:
    import galois  # noqa: F401
    GALOIS_FOUND = True
except ImportError:
    GALOIS_FOUND = False


def _cuda_is_available():
    if shutil.which("nvcc") is None:
        return False

    try:
        import pycuda.driver as cuda
        cuda.init()
        return cuda.Device.count() > 0
    except Exception:
        return False


CUDA_FOUND = _cuda_is_available()
