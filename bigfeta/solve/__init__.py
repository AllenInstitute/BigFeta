from . import solve_scipy
from . import solve_petsc

# uses petsc if available.  Can set default solve by modifying this.
default_solve = (solve_petsc.solve
                 if solve_petsc.HAS_PETSC
                 else solve_scipy.solve)


def solve(*args, **kwargs):
    # py2 compatible version of kw-only arg
    solve_func = kwargs.pop("solve_func", default_solve)
    return solve_func(*args, **kwargs)


solve_funcs = {
    "petsc": solve_petsc.solve,
    "scipy": solve_scipy.solve,
    "default": default_solve
}

__all__ = ["solve", "solve_funcs"]
