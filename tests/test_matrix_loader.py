import pytest
import numpy

from syngular import Field, Ring, RingPoint

from linac import load_matrices, cuda_row_reduce


def f(x, y, z):
    return (
        (x * (x + 2 * y + 3 * z + 5 * x * y * z)) / (x ** 2 + y + z) +
        (y * (x ** 2 + 7 * x * y + 11 * y * z)) / (x + y + z ** 2)
    )


prefactors = ['(x)/(x^2+y+z)', '(y)/(x+y+z^2)', '(z)/(x+y^2+z)']

ansatze = [
    [['x'], ['y'], ['z'], ['x', 'x'], ['x', 'y'], ['x', 'z'], ['y', 'y'], ['y', 'z'], ['x', 'y', 'z']],
    [['x'], ['y'], ['z'], ['x', 'x'], ['x', 'y'], ['x', 'z'], ['y', 'y'], ['y', 'z'], ['x', 'y', 'z']],
    [],
]

ring = Ring('0', ('x', 'y', 'z'), 'dp')


@pytest.mark.parametrize(
    "field", (Field("mpf", 0, 64), Field("mpc", 0, 64),
              Field("finite field", 2 ** 31 - 1, 1), Field("padic", 2 ** 31 - 1, 5),
              Field("finite field", 2 ** 31 - 19, 1), Field("finite field", 2 ** 31 + 11, 1),
              Field("finite field", 2 ** 61 - 1, 1), Field("padic", 2 ** 61 - 1, 5),
              Field("finite field", 2 ** 61 + 15, 1), )
)
def test_matrix_loader_and_sys_solver(field):
    nInputs = sum(map(len, ansatze))
    points = [RingPoint(ring, field, seed=seed) for seed in range(nInputs)]
    As = AsGPU = load_matrices(prefactors, ansatze, points)
    AsCPU = load_matrices(prefactors, ansatze, points, use_cuda=False)
    if field.characteristic == 0:
        assert numpy.isclose([numpy.max(A1 - A2, initial=0) for A1, A2 in zip(AsGPU, AsCPU)], [0, 0, 0]).all()
    else:
        assert all([numpy.all(A1 == A2) for A1, A2 in zip(AsGPU, AsCPU)])
    b = numpy.array([f(*point.values()) for point in points])
    if field.name == "padic":
        b = numpy.vectorize(int)(b) % field.characteristic
    b = numpy.atleast_2d(b).T
    A = numpy.block(As)
    Ab = numpy.block([A, b])
    rref = cuda_row_reduce(Ab, field_characteristic=field.characteristic)
    if field.characteristic == 0:
        assert numpy.isclose(rref[:, :-1], numpy.identity(Ab.shape[0], int)).all()
    else:
        assert numpy.all(rref[:, :-1] == numpy.identity(Ab.shape[0], int))
    if field.characteristic == 0:
        assert numpy.isclose(rref[:, -1].tolist(), [1, 2, 3, 0, 0, 0, 0, 0, 5, 0, 0, 0, 1, 7, 0, 0, 11, 0]).all()
    else:
        rref[:, -1].tolist() == [1, 2, 3, 0, 0, 0, 0, 0, 5, 0, 0, 0, 1, 7, 0, 0, 11, 0]
