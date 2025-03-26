import functools
import operator
import numpy


def memoized(*decorator_args, **decorator_kwargs):
    """Diskcaching decorator generator."""
    def memoized_decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self = args[0]

            if not hasattr(self, 'diskcache'):
                return func(*args, **kwargs)

            @self.diskcache.memoize(*decorator_args, **decorator_kwargs)
            def diskcached_func(*args, **kwargs):
                return func(*args, **kwargs)
            return diskcached_func(*args, **kwargs)
        return wrapper
    return memoized_decorator


class tensor_function(object):
    """Tensor function supporting indexing and iteration.
       For instance, the initializer 'callable_function' can be a function returning a numpy.array"""

    def __init__(self, callable_function):
        self.callable_function = callable_function
        if hasattr(callable_function, "__name__"):
            self.__name__ = callable_function.__name__

    def flatten(self):
        selfFlattened = tensor_function(lambda args: self(args).flatten())
        if hasattr(self, '__shape__'):
            selfFlattened.shape = (functools.reduce(operator.mul, self.__shape__), )
        return selfFlattened

    def __getitem__(self, index):
        return tensor_function(lambda args: self(args)[index])

    def __matmul__(self, other):
        assert isinstance(other, numpy.ndarray)
        return tensor_function(lambda args: self(args) @ other)

    @memoized(name='tensor_function.__call__', ignore={0})
    def __call__(self, *args, **kwargs):
        res = self.callable_function(*args, **kwargs)
        if hasattr(res, "shape") and not hasattr(self, "__shape__"):
            self.shape = res.shape
        elif isinstance(res, list):
            self.__size__ = len(res)
        return res

    def __len__(self):
        if hasattr(self, "shape"):
            return self.shape[0]
        elif hasattr(self, "__size__"):
            return self.__size__
        else:
            raise AttributeError("Length not known. Have you tried evaluating the function at least once?")

    @property
    def shape(self):
        if hasattr(self, "__shape__"):
            return self.__shape__
        else:
            raise AttributeError("Shape not known. Have you tried evaluating the function at least once?")

    @shape.setter
    def shape(self, value):
        self.__shape__ = value

    def __iter__(self):
        selfFlattened = self.flatten()
        for i in range(selfFlattened.shape[0]):
            yield self[i]
