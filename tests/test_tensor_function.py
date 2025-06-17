import numpy

from linac import tensor_function


def fs(x, y):
    return numpy.array(
        [[x, y, x + y, x + 3 * y],
         [x ** 2, y ** 2, x * y, x ** 2 + 2 * x * y + y ** 2]]
    )


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
