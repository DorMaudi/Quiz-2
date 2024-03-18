"""
Date: 18/3/2024
Group:
        Sapir Natanov 322378068
        Dor Maudi 207055138
        Noa Yasharzadeh 208595157
        Segev Isaac 207938085
Git: https://github.com/DorMaudi/team-project-1
Name: Dor Maudi 207055138
"""

import numpy as np
from numpy.linalg import norm
from colors import bcolors
from matrix_utility import is_diagonally_dominant, DominantDiagonalFix


def gauss_seidel(A, b, X0, TOL=1e-16, N=201):
    n = len(A)
    k = 1

    if is_diagonally_dominant(A):
        print('Matrix is diagonally dominant - preforming jacobi algorithm\n')
    else:
        A, b = DominantDiagonalFix(A, b)
        if isinstance(A, list):
            print('Matrix is diagonally dominant - preforming jacobi algorithm\n')
        else:
            print('Matrix is NOT diagonally dominant\n')
            return None

    print(
        "Iteration" + "\t\t\t".join([" {:>12}".format(var) for var in ["x{}".format(i) for i in range(1, len(A) + 1)]]))
    print("-----------------------------------------------------------------------------------------------")
    x = np.zeros(n, dtype=np.double)
    while k <= N:

        for i in range(n):
            sigma = 0
            for j in range(n):
                if j != i:
                    sigma += A[i][j] * x[j]
            x[i] = (b[i] - sigma) / A[i][i]

        print("{:<15} ".format(k) + "\t\t".join(["{:<15} ".format(val) for val in x]))

        if norm(x - X0, np.inf) < TOL:
            return tuple(x)

        k += 1
        X0 = x.copy()

    print("Maximum number of iterations exceeded - no solution found")
    # return tuple(x)


if __name__ == '__main__':
    A = np.array([[2, 3, 4, 5, 6],
                  [-5, 3, 4, -2, 3],
                  [4, -5, -2, 2, 6],
                  [4, 5, -1, -2, -3],
                  [5, 5, 3, -3, 5]])

    b = np.array([92, 22, 42, -22, 41])

    X0 = np.zeros_like(b)

    solution = gauss_seidel(A, b, X0)
    print(bcolors.OKBLUE, "\nApproximate solution:", solution)
