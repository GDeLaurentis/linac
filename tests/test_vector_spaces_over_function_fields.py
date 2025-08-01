import numpy

from syngular import Field

from linac import tensor_function, VectorSpaceOfFunctions


def fs(*args):
    x, y = args[0]
    return numpy.array(
        [[x, y, x + y, x + 3 * y],
            [x ** 2, y ** 2, x * y, x ** 2 + 2 * x * y + y ** 2]]
    )


oFs = tensor_function(fs)
oFs = oFs.flatten()


def test_VS_of_bivariate_polys():
    field = Field('finite field', 2147483647, 1)
    oVs = VectorSpaceOfFunctions(oFs, lambda seed: (field.random(), field.random()), field=field,
                                 iteration_step=2, max_iteration=4,)
    assert oVs.dim == 5
    assert oVs.pivots == [0, 1, 4, 5, 6]


def test_VS_kernel():
    field = Field('finite field', 2147483647, 1)
    oVs = VectorSpaceOfFunctions(oFs, lambda seed: (field.random(), field.random()), field=field)
    assert oVs.kernel.shape == (8, 3)
    assert numpy.all(oVs.kernel == numpy.array(
        [[1, 1, 0],
         [1, 3, 0],
         [-1, 0, 0],
         [0, -1, 0],
         [0, 0, 1],
         [0, 0, 1],
         [0, 0, 2],
         [0, 0, -1]])
    )


def test_VS_contains_function():
    field = Field('finite field', 2147483647, 1)
    oVs = VectorSpaceOfFunctions(oFs, lambda seed: (field.random(), field.random()), field=field)
    assert oFs[0] in oVs


def test_VS_contains_VS():
    field = Field('finite field', 2147483647, 1)
    oVs = VectorSpaceOfFunctions(oFs, lambda seed: (field.random(), field.random()), field=field)
    oSubVS = VectorSpaceOfFunctions(oFs[:2], lambda seed: (field.random(), field.random()), field=field)
    assert oSubVS in oVs
    assert oVs not in oSubVS


def test_VS_add():
    field = Field('finite field', 2147483647, 1)
    oVs = VectorSpaceOfFunctions(oFs, lambda seed: (field.random(), field.random()), field=field)
    oVs2 = VectorSpaceOfFunctions(oFs[:4], lambda seed: (field.random(), field.random()), field=field)
    oVs3 = VectorSpaceOfFunctions(oFs[4:], lambda seed: (field.random(), field.random()), field=field)
    oVsX = oVs2 + oVs3
    assert oVs in oVsX and oVsX in oVs
