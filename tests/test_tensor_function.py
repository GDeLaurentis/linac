import diskcache
import time
import random
import numpy

from linac import tensor_function


def fs(x, y):
    return numpy.array(
        [[x, y, x + y, x + 3 * y],
         [x ** 2, y ** 2, x * y, x ** 2 + 2 * x * y + y ** 2]]
    )


def gs(x, y):
    return [x, y, x + y, x + 3 * y]


def slow_function(*args, **kwargs):
    time.sleep(1)
    return numpy.array([1, 2])


oSlowFunc = tensor_function(slow_function)
oSlowFunc.diskcache = diskcache.Cache(
    directory=f"/tmp/linac/tensor_function_diskcache_test_{random.randint(0, 10 ** 9)}",
    size_limit=10 * 2 ** 30
)


def test_tf_plain_list():
    oFs = tensor_function(gs)
    oFs(1, 2)
    assert len(oFs) == 4


def test_tf_evaluation():
    oFs = tensor_function(fs)
    assert numpy.all(oFs(1, 2) == fs(1, 2))


def test_tf_flatten():
    oFs = tensor_function(fs)
    assert numpy.all(oFs.flatten()(1, 2) == oFs(1, 2).flatten())


def test_tf_slicing():
    oFs = tensor_function(fs)
    assert numpy.all(oFs[:2, :2](1, 2) == oFs(1, 2)[:2, :2])


def test_tf_matmul():
    oFs = tensor_function(fs)
    assert numpy.all(
        oFs(1, 2) @ numpy.array(
            [[0, 1, 0, 0],
             [1, 0, 0, 0],
             [0, 0, 0, 1],
             [0, 0, 1, 0]]
        ) == (oFs @ numpy.array(
            [[0, 1, 0, 0],
             [1, 0, 0, 0],
             [0, 0, 0, 1],
             [0, 0, 1, 0]]
        ))(1, 2)
    )


def test_tf_shape_and_length():
    oFs = tensor_function(fs)
    oFs(1, 2)
    assert oFs.shape == (2, 4)
    assert len(oFs) == 2


def test_tf_iter():
    oFs = tensor_function(fs)
    oFs(1, 2)
    assert numpy.all(oFs.flatten()(1, 2) == numpy.array([oF(1, 2) for oF in oFs]))


def test_caching_speeds_up():
    # Uncached version
    start_uncached = time.perf_counter()
    result_uncached = oSlowFunc(42)
    end_uncached = time.perf_counter()
    time_uncached = end_uncached - start_uncached

    # Cached version
    start_cached = time.perf_counter()
    result_cached = oSlowFunc(42)
    end_cached = time.perf_counter()
    time_cached = end_cached - start_cached

    assert numpy.all(result_uncached == result_cached), "Results differ!"
    assert time_cached < time_uncached, f"Caching didn't speed things up: uncached={time_uncached:.6f}s, cached={time_cached:.6f}s"
