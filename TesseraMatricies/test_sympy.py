import numpy as np
import sympy as sym
import scipy, scipy.io
from sympy import Pow as pow
x = sym.Symbol('x')
y = sym.Symbol('y')
z = sym.Symbol('z')

def evaluate_string(exp_str):
    
    exp_pow = exp_str.replace("^", "**")
    expression, = sym.parse_expr(exp_pow, evaluate = True)
    expression = sym.simplify(expression)
    evalf_exp = expression.evalf()
    return evalf_exp

def evaluate_matrix(str_mat):
    shape = str_mat.shape
    evalf_mat = str_mat.copy()
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            print("### INPUT ###", i,j, str_mat[i,j])
            evalf_exp = evaluate_string(str_mat[i,j])
            print("=== OUTPUT ===", evalf_exp)
            evalf_mat[i,j] = evalf_exp
    return evalf_mat

def remove_power_in_matrix(mat):
    shape = mat.shape
    new_mat = mat.copy()
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            print("### INPUT ###", i,j, mat[i,j])
            no_pow = mat[i,j].replace("x**", "x")
            no_pow = mat[i,j].replace("y**", "y")
            no_pow = mat[i,j].replace("z**", "z")
            print("=== OUTPUT ===", no_pow)
            new_mat[i,j] = no_pow
    return new_mat


if __name__ == "__main__":
    mat = np.loadtxt("matrixForTesseral.mat", dtype = "str")
    print(mat.shape)
    evalf_mat = evaluate_matrix(mat)
    no_pow_mat = remove_power_in_matrix(evalf_mat)
    np.savetxt("tesseral_mat_nopow.txt", no_pow_mat, fmt="%s")
