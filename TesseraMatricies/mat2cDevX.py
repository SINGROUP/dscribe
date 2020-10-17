import numpy as np; import sympy as sym; from sympy import Pow as pow
x = sym.Symbol('x'); y = sym.Symbol('y'); z = sym.Symbol('z')

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
#            if (str_mat[i,j] != "0,"):
#            print("### INPUT ###", i,j, str_mat[i,j])
            evalf_exp = evaluate_string(str_mat[i,j])
            print(str_mat[i])
            print("=== OUTPUT ===", evalf_exp)
            evalf_mat[i,j] = evalf_exp
    return evalf_mat

def remove_power_in_matrix(mat):
    shape = mat.shape
    new_mat = mat.copy()
    
    for i in range(shape[0]):
        for j in range(shape[1]):
#            print(type(str_mat[i,j]), str_mat[i,j])
#            if (str_mat[i,j] != " 0,"):
#                print("### INPUT ###", i,j, mat[i,j])
            no_powX = mat[i,j].replace("x**", "x")
            no_powXY = no_powX.replace("y**", "y")
            no_powXYZ = no_powXY.replace("z**", "z")
            no_powXYZR2 = no_powXYZ.replace("(x2 + y2 + z2)", "rr")
            no_powXYZR2Final = no_powXYZR2.replace("**","" )
            print(no_powXYZR2Final,"X", "\n")
            print("=== OUTPUT ===", mat[i,j])
            new_mat[i,j] = no_powXYZR2Final
    return new_mat



if __name__ == "__main__":
    # dtype large so that string isn't cut off
    mat = np.loadtxt("tesseralMatDerX.mat", dtype = "U16384")
#    mat = np.loadtxt("test.txt", dtype = "U16384") # Change me
    print(mat.shape)
    evalf_mat = evaluate_matrix(mat)
    no_pow_mat = remove_power_in_matrix(evalf_mat)
    np.savetxt("tesseral_mat_nopowDevX.txt", no_pow_mat, fmt="%s",delimiter=", ")
    np.save("tesseral_mat_nopowDevX.npy", no_pow_mat)
