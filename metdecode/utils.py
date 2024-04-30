import argparse
from typing import Callable

import numpy as np


def bounded_float_type(lb: float = -np.inf, ub: float = np.inf) -> Callable:
    def _bounded_float_type(arg: str):
        try:
            f = float(arg)
        except ValueError:
            raise argparse.ArgumentTypeError('Must be a floating point number')
        if f < lb or f > ub:
            raise argparse.ArgumentTypeError(f'Argument must be in the range [{lb}, {ub}]')
        return f
    return _bounded_float_type
