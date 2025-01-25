import sympy as sp
import time
from functools import wraps
from enum import Enum

from libringkit import nullspace as rust_nullspace, qr as rust_qr


class DType(Enum):
    BigInt = 0
    BigRational = 1


def time_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f'{func.__name__} took {time.time() - start}')
        return result
    return wrapper


def sp_to_vec(A, dtype):
    @time_it
    def get_lcm():
        return sp.lcm([
            sp.denom(A[i, j]) for i in range(
                A.rows) for j in range(
                    A.cols) if not A[i, j].is_zero])
        
    A_lcm = sp.sympify(1) if dtype == DType.BigRational else get_lcm()

    def choose_conversion():
        if dtype == DType.BigRational:
            return lambda elem: (int(sp.numer(elem)), int(sp.denom(elem)))
        return lambda elem: int(sp.cancel(A_lcm * elem)) if not elem.is_zero else 0
    
    to_elem = choose_conversion()

    @time_it
    def do_convert():
        return [to_elem(A[row, col]) for row in range(A.rows) for col in range(A.cols)]

    return do_convert(), A_lcm


@time_it
def timed_sp_to_vec(A, dtype):
    return sp_to_vec(A, dtype)


def vec_to_sp(A, dtype):
    def choose_conversion():
        if dtype == DType.BigRational:
            return lambda row: [sp.Rational(sp.S(numer), sp.S(denom)) for (numer, denom) in row]
        return lambda row: [sp.S(elem) for elem in row]
    
    to_row = choose_conversion()

    return sp.Matrix([to_row(row) for row in A])


@time_it
def timed_vec_to_sp(A, dtype):
    return vec_to_sp(A, dtype)

@time_it
def timed_rust_nullspace(rows, cols, A, dtype):
    return rust_nullspace(rows, cols, A, dtype.value)

def nullspace(A, dtype):
    A_np, _ = timed_sp_to_vec(A, dtype)

    basis = timed_rust_nullspace(A.rows, A.cols, A_np, dtype)

    basis_mat = timed_vec_to_sp(basis, dtype)

    return basis_mat


def qr(A, dtype):
    A_typed, A_lcm = sp_to_vec(A, dtype)

    Q_np, R_np = rust_qr(A.rows, A.cols, A_typed, dtype.value)

    Q_sp = vec_to_sp(Q_np, dtype)
    R_sp = vec_to_sp(R_np, dtype)

    if dtype != DType.BigRational:
        R_sp = R_sp.applyfunc(lambda x: sp.cancel(sp.Rational(x, A_lcm)))

    return Q_sp, R_sp
