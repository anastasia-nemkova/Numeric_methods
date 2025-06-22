import math
import numpy as np

def read_file(filename):
    with open(filename, "r") as file:
        lines = file.readlines()
    A = []
    for line in lines[:-1]:
        A.append(list(map(float, line.split())))
    
    eps = float(lines[-1])
    return A, eps

def write_in_file(filename, A, eps, eigenvalues, eigenvectors, iter):
    with open(filename, "w") as file:
        file.write("Matrix A:\n")
        for row in A:
            file.write(" ".join(f"{val:.2f}" for val in row) + "\n")
        
        file.write(f"\nEps: {eps}\n")
        
        file.write("\nResult rotation method:\n")
        file.write("Eigenvalues:\n")
        file.write(" ".join(f"{val:.6f}" for val in eigenvalues) + "\n")
        file.write("\nEigenvectors:\n")
        for i, col in enumerate(zip(*eigenvectors)):
            file.write(f"vec{i+1} : [" + ", ".join(f"{val:.6f}" for val in col) + "]\n")
        file.write(f"\nIterations: {iter}\n")
        
        file.write("\nCheck with numpy:\n")
        np_eigenvalues, np_eigenvectors = np.linalg.eig(A)
        file.write("Eigenvalues:\n")
        file.write(" ".join(f"{val:.6f}" for val in np_eigenvalues) + "\n")
        file.write("\nEigenvectors:\n")
        for i, row in enumerate(np_eigenvectors.T):
            file.write(f"vec{i+1} : [" + ", ".join(f"{val:.6f}" for val in row) + "]\n")     

def max_off_diag_elem(A):
    # Нахождение максимального по модулю недиагонального элемента
    n = len(A)
    max_elem = 0
    i_max = 0
    j_max = 1
    for i in range(n):
        for j in range(i + 1, n):
            if abs(A[i][j]) > max_elem:
                max_elem = abs(A[i][j])
                i_max = i
                j_max = j
    return i_max, j_max

def get_phi(a_ll, a_mm, a_lm):
    # Вычисление угла поворота
    if a_ll == a_mm:
        return math.pi / 4
    else:
        return 0.5 * math.atan(2 * a_lm / (a_ll - a_mm))

def normal_mat(A):
    n = len(A)
    norm = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            norm += A[i][j] * A[i][j]
            
    return math.sqrt(norm)

def rotate_mat(A, i, j):
    n = len(A)
    P = [[1 if k == l else 0 for l in range(n)] for k in range(n)]
    
    phi = get_phi(A[i][i], A[j][j], A[i][j])
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)
    
    P[i][i] = P[j][j] = cos_phi
    P[i][j] = -sin_phi
    P[j][i] = sin_phi
    return P

def transpose(A):
    n = len(A)
    return [[A[j][i] for j in range(n)] for i in range(n)] 

def matrix_multiply(A, B):
    rows_A = len(A)
    cols_A = len(A[0])
    cols_B = len(B[0])
    result = [[0.0 for _ in range(cols_B)] for _ in range(rows_A)]

    for i in range(rows_A):
        for j in range(cols_B):
            result[i][j] = sum(A[i][k] * B[k][j] for k in range(cols_A))
    
    return result

def rotation_method(A, eps, max_iter=100):
    n = len(A)
    V = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    iter = 0
    
    while iter < max_iter:
        i, j = max_off_diag_elem(A)
        if normal_mat(A) < eps:
            break
        
        P = rotate_mat(A, i, j)
        P_T = transpose(P)
        A = matrix_multiply(matrix_multiply(P_T, A), P)
        V = matrix_multiply(V, P)
        
        iter += 1
        
    eigenvalues = [A[i][i] for i in range(n)]
    return eigenvalues, V, iter
        
    
def main():
    A, eps = read_file("input.txt")

    eigenvalues, eigenvectors, iter = rotation_method(A, eps)
    
    write_in_file("output.txt", A, eps, eigenvalues, eigenvectors, iter)

    
if __name__ == "__main__":
    main()