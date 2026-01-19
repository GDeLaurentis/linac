try:
    import galois  # noqa: F401
    GALOIS_FOUND = True
except ImportError:
    GALOIS_FOUND = False
