
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
from matrix_utility import swap_rows_elementary_matrix, row_addition_elementary_matrix


def lu(A):
    N = len(A)
    L = np.eye(N)  # Create an identity matrix of size N x N

    for i in range(N):

        # Partial Pivoting: Find the pivot row with the largest absolute value in the current column
        # pivot_row = i
        # v_max = abs(A[pivot_row][i])
        # for j in range(i + 1, N):
        #     if abs(A[j][i]) > v_max:
        #         v_max = abs(A[j][i])
        #         pivot_row = j

        pivot_row = i
        v_max = abs(A[pivot_row][i])
        for j in range(i + 1, N):
            if abs(A[j][i]) > v_max:
                v_max = abs(A[j][i])
                pivot_row = j

        # Swap the current row with the pivot row
        if pivot_row != i:
            e_matrix = swap_rows_elementary_matrix(N, i, pivot_row)
            print(f"elementary matrix for swap between row {i} to row {pivot_row} :\n {e_matrix} \n")
            A = np.dot(e_matrix, A)
            print(f"The matrix after elementary operation :\n {A}")
            print(bcolors.OKGREEN, "---------------------------------------------------------------------------",
                  bcolors.ENDC)

            # if a principal diagonal element is zero,it denotes that matrix is singular,
            # and will lead to a division-by-zero later.
            if A[i][i] == 0:
                raise ValueError("can't perform LU Decomposition")

        for j in range(i + 1, N):
            #  Compute the multiplier
            m = -A[j][i] / A[i][i]
            e_matrix = row_addition_elementary_matrix(N, j, i, m)
            e_inverse = np.linalg.inv(e_matrix)
            L = np.dot(L, e_inverse)
            A = np.dot(e_matrix, A)
            print(f"elementary matrix to zero the element in row {j} below the pivot in column {i} :\n {e_matrix} \n")
            print(f"The matrix after elementary operation :\n {A}")
            print(bcolors.OKGREEN, "---------------------------------------------------------------------------",
                  bcolors.ENDC)

    U = A
    return L, U


# function to calculate the values of the unknowns
def backward_substitution(A_b, U, L):
    N = len(U)
    b = np.zeros(N)

    for i in range(N):
        b[i] = A_b[i][N]

    Umat = np.zeros((N, N))
    Lmat = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            Umat[i][j] = U[i][j]
            Lmat[i][j] = L[i][j]

    UinvFinal = np.linalg.inv(Umat)
    LinvFinal = np.linalg.inv(Lmat)

    print(bcolors.OKGREEN, np.dot(Lmat, Umat), bcolors.ENDC)

    sol = np.dot(np.dot(UinvFinal, LinvFinal), b)

    return sol


def lu_solve(A_b):
    L, U = lu(A_b)
    OgMat = np.dot(L, U)
    print("Lower triangular matrix L:\n", L)
    print("Upper triangular matrix U:\n", U)

    result = backward_substitution(OgMat, U, L)
    print(bcolors.OKBLUE, "\nSolution for the system:")
    for x in result:
        print("{:.6f}".format(x))


if __name__ == '__main__':
    # A_b = [[1, 2, 1, 2],
    #        [2, 6, 1, 4],
    #        [1, 1, 4, 8]]

    A_b = [[0, 1, 1, 1, -8],
           [1, 1, 2, 1, -20],
           [2, 2, 4, 0, -2],
           [1, 2, 1, 1, 4]]

    lu_solve(A_b)
