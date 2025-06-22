def read_matrix(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    A = []
    for line in lines[:-1]:
        A.append(list(map(float, line.split())))
    
    b = list(map(float, lines[-1].split()))
    return A, b


def write_in_file(filename, A, b, L, U, x, det_A, inverse):
    with open(filename, "w") as file:
        file.write("Matrix A:\n")
        for row in A:
            file.write(" ".join(f"{val:.8f}" for val in row) + "\n")
        
        file.write("\nVector b:\n")
        file.write(" ".join(f"{val:.8f}" for val in b) + "\n")
        
        file.write("\nMatrix L:\n")
        for row in L:
            file.write(" ".join(f"{val:.8f}" for val in row) + "\n")
            
        file.write("\nMatrix U:\n")
        for row in U:
            file.write(" ".join(f"{val:.8f}" for val in row) + "\n")
            
        file.write("\nSolve x:\n")
        file.write(" ".join(f"{val:.8f}" for val in x) + "\n")
            
        file.write("\nDeterminant A: " + str(det_A) + "\n")
        
        file.write("\nInverse matrix A^(-1):\n")
        for row in inverse:
            file.write(" ".join(f"{val:.8f}" for val in row) + "\n")
            
    checks(filename, x, A, b, inverse)

def pivotize(A):
    n = len(A)
    P = [[float(i == j) for i in range(n)] for j in range(n)]
    
    for i in range(n):
        max_elem = max(range(i, n), key=lambda j: abs(A[j][i]))
        if i != max_elem:
            P[i], P[max_elem] = P[max_elem], P[i]
    return P

def matrix_multypay(A, B):
    result = [[0.0] * len(B[0]) for _ in range(len(A))]
    
    for i in range(len(A)):
        for j in range(len(B[0])):
            sum_value = 0
            for k in range(len(B)):
                sum_value += A[i][k] * B[k][j]
            result[i][j] = sum_value
    
    return result

def lu_decomposition(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]
    P = pivotize(A)
    PA = matrix_multypay(P, A)
    
    for j in range(n):
        L[j][j] = 1.0
        for i in range(j + 1):
            sum_value = 0
            for k in range(i):
                sum_value += U[k][j] * L[i][k]
            U[i][j] = PA[i][j] - sum_value
        
        for i in range(j, n):
            sum_value = 0
            for k in range(j):
                sum_value += U[k][j] * L[i][k]
            L[i][j] = (PA[i][j] - sum_value) / U[j][j]
    
    return L, U, P


def solve_slay(L, U, P, b):
    n = len(L)
    Pb = [sum(P[i][j] * b[j] for j in range(n)) for i in range(n)]
    
    # Прямой ход (Lz = Pb)
    z = [0.0 for _ in range(n)]
    for i in range(n):
        z[i] = Pb[i]
        for j in range(i):
            z[i] -= L[i][j] * z[j]
    
    # Обратный ход (Ux = z)
    x = [0.0 for _ in range(n)]
    for i in range(n - 1, -1, -1):
        x[i] = z[i] / U[i][i]
        for j in range(i + 1, n):
            x[i] -= U[i][j] * x[j] / U[i][i]
    
    return x

def determinant(U):
    """
    A = LU
    det(A) = det(LU)
    det(A) = det(L)*det(U)
    det(L) = 1
    det(A) = det(U)
    """
    det = 1.0
    for i in range(len(U)):
        det *= U[i][i]
    return det

def inverse_matrix(A):
    """
    A * A^(-1) = E
    LU * x = E
    """
    n = len(A)
    A1 = [[0.0 for _ in range(n)] for _ in range(n)]
    
    L, U, P = lu_decomposition(A)
    
    for i in range(n):
        e = [1 if j == i else 0 for j in range(n)]
        x = solve_slay(L, U, P, e)
        for j in range(n):
            A1[j][i] = x[j]
    
    return A1

def checks(filename, x, A, b, inv_A):
    with open(filename, "a") as file:
        file.write("\nChecks:\n")
        
        
        # Verify that A * x = b
        Ax = [sum(A[i][j] * x[j] for j in range(len(A[i]))) for i in range(len(A))]
        file.write("\n1) Result of A * x:\n")
        for value in Ax:
            file.write(f"{value:.8f} ")
        file.write("\nOriginal b:\n")
        for value in b:
            file.write(f"{value} ")
        file.write("\n")
        
        is_solution_correct = True
        for i in range(len(b)):
            if not (abs(Ax[i] - b[i]) < 1e-6): 
                is_solution_correct = False
                break
        if is_solution_correct:
            file.write("The solution x satisfies A * x = b.\n")
        else:
            file.write("The solution x does not satisfy A * x = b.\n")

        # Verify that A * A^(-1) is the identity matrix
        identity_check = True
        identity_matrix = [[float(i == j) for i in range(len(A))] for j in range(len(A))]
        A_inv_mult = matrix_multypay(A, inv_A)
        
        file.write("\n2) Result of A * A^(-1):\n")
        for row in A_inv_mult:
            file.write(" ".join(f"{val:.8f}" for val in row) + "\n")
        
        for i in range(len(A)):
            for j in range(len(A)):
                if abs(A_inv_mult[i][j] - identity_matrix[i][j]) > 1e-6:
                    identity_check = False
                    break
            if not identity_check:
                break
        
        if identity_check:
            file.write("A * A^(-1) is the identity matrix.\n")
        else:
            file.write("A * A^(-1) is not the identity matrix.\n")

def main():
    A, b = read_matrix("input.txt")
    A_orig = [row[:] for row in A]
    
    L, U, P = lu_decomposition(A)
    det_A = determinant(U)
    x = solve_slay(L, U, P, b)
    inv_A = inverse_matrix(A_orig)
    
    write_in_file("output.txt", A_orig, b, L, U, x, det_A, inv_A)

if __name__ == "__main__":
    main()