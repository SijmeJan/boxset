import numpy as np
from numba import jit

from numba.np.unsafe.ndarray import to_fixed_tuple, tuple_setitem


@jit
def my_swapaxes(a, axis1, axis2):
    """ Numba swapaxes ala np.swapaxes() """
    axis1 = a.ndim + axis1 if axis1 < 0 else axis1
    axis2 = a.ndim + axis2 if axis2 < 0 else axis2
    if axis1 == axis2:
        return a
    pos = to_fixed_tuple(np.arange(a.ndim), a.ndim)
    tmp = pos[axis1]
    pos = tuple_setitem(pos, axis1, pos[axis2])
    pos = tuple_setitem(pos, axis2, tmp)
    return a.transpose(pos)
