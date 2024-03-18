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
from colors import bcolors
from matrix_utility import swap_row


def gaussianElimination(mat):
    N = len(mat)

    singular_flag = forward_substitution(mat)

    if singular_flag != -1:

        if mat[singular_flag][N]:
            return "Singular Matrix (Inconsistent System)"
        else:
            return "Singular Matrix (May have infinitely many solutions)"

    # if matrix is non-singular: get solution to system using backward substitution
    return backward_substitution(mat)


def forward_substitution(mat):
    N = len(mat)
    for k in range(N):

        # Partial Pivoting: Find the pivot row with the largest absolute value in the current column
        pivot_row = k
        v_max = mat[pivot_row][k]
        for i in range(k + 1, N):
            if abs(mat[i][k]) > v_max:
                v_max = abs(mat[i][k])
                pivot_row = i

        # if a principal diagonal element is zero,it denotes that matrix is singular,
        # and will lead to a division-by-zero later.
        if not mat[pivot_row][k]:
            return k  # Matrix is singular

        # Swap the current row with the pivot row
        if pivot_row != k:
            swap_row(mat, k, pivot_row)
        # End Partial Pivoting

        for u in range(N):
            if not round(mat[u][u], 4):
                mat[u][u] = 0
                print(f"matrix: \n{np.array(mat)}\n")
                return N - 1

        for z in range(k, N):
            mat[k][z + 1] /= mat[k][k]
        mat[k][k] /= mat[k][k]

        for i in range(k + 1, N):

            #  Compute the multiplier
            m = mat[i][k] / mat[k][k]
            # if mat[k][k] != 0:
            #     mat[k][k] /= mat[k][k]
            # else:
            #     raise ValueError("Matrix is singular.\n")

            # subtract fth multiple of corresponding kth row element
            for j in range(k + 1, N + 1):
                mat[i][j] -= mat[k][j] * m

            # filling lower triangular matrix with zeros
            mat[i][k] = 0
            print(f"matrix: \n{np.array(mat)}\n")

        # for z in range(k, N):
        #     mat[k][z+1] /= mat[k][k]
        # mat[k][k] /= mat[k][k]

    mat[N - 1][N] /= mat[N - 1][N - 1]
    mat[N - 1][N - 1] /= mat[N - 1][N - 1]
    print(f"matrix: \n{np.array(mat)}\n")
    return -1


# function to calculate the values of the unknowns
def backward_substitution(mat):
    N = len(mat)
    x = np.zeros(N)  # An array to store solution

    # zero out upper triangular.
    for k in range(N - 1, -1, -1):
        for b in range(k - 1, -1, -1):

            #  Compute the multiplier
            m = mat[b][k]

            # subtract fth multiple of corresponding kth row element
            for j in range(k, N + 1):
                if j != k:
                    mat[b][j] -= mat[k][j] * m
                else:
                    mat[b][j] -= mat[b][j] * mat[k][j]

            # filling lower triangular matrix with zeros
            # mat[b][k] = 0
            print(f"matrix: \n{np.array(mat)}\n")

    # Start calculating from last equation up to the first
    for i in range(N - 1, -1, -1):

        x[i] = mat[i][N]

        # Initialize j to i+1 since ma.rix is upper triangular
        for j in range(i + 1, N):
            x[i] -= mat[i][j] * x[j]

        x[i] = (x[i] / mat[i][i])

    return x


if __name__ == '__main__':

    A_b = [[2, 3, 4, 5, 6, 92],
           [-5, 3, 4, -2, 3, 22],
           [4, -5, -2, 2, 6, 42],
           [4, 5, -1, -2, -3, -22],
           [5, 5, 3, -3, 5, 41]]

    np.set_printoptions(suppress=True, precision=4)

    result = gaussianElimination(A_b)
    if isinstance(result, str):
        print(result)
    else:
        print(bcolors.OKBLUE, "\nSolution for the system:")
        for x in result:
            print("{:.6f}".format(x))
