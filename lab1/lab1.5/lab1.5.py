import cmath
import numpy as np

def read_file(filename):
    with open(filename, "r") as file:
        lines = file.readlines()
    A = []
    for line in lines[:-1]:
        A.append(list(map(float, line.split())))
    
    eps = float(lines[-1])
    return A, eps

def write_in_file(filename, A, eps, Q, R, H, eigenval, iter, A_i, A_92):
    with open(filename, "w") as file:
        file.write("Matrix A:\n")
        for row in A:
            file.write(" ".join(f"{val:.6f}" for val in row) + "\n")
        
        file.write(f"\nEps: {eps}\n")
        
        file.write("\nMatrix Q:\n")
        for row in Q:
            file.write(" ".join(f"{val:.6f}" for val in row) + "\n")
        
        file.write("\nMatrix R:\n")
        for row in R:
            file.write(" ".join(f"{val:.6f}" for val in row) + "\n")
        
        file.write("\nHouseholder Matrices:\n")
        for i, H in enumerate(H):
            file.write(f"Householder Matrix {i+1}:\n")
            for row in H:
                file.write(" ".join(f"{val:.6f}" for val in row) + "\n")
            file.write("\n")
            
        file.write("Matrix A 92:\n")
        for row in A_92:
            file.write(" ".join(f"{val:.6f}" for val in row) + "\n")
            
        file.write("Matrix A result:\n")
        for row in A_i:
            file.write(" ".join(f"{val:.6f}" for val in row) + "\n")
            
        file.write("\nEigenvalues:\n")
        for val in eigenval:
            file.write(f"{val:.6f}\n")
            
        file.write(f"\nIterations: {iter}\n")
                
        file.write("\nCheks with numpy:\n")
        file.write("Eigenvalues:\n")
        expected_eigenvalues = np.linalg.eigvals(A)
        for eigenvalue in expected_eigenvalues:
            file.write(f"{eigenvalue:.6f}\n")

def matrix_multiply(A, B):
    rows_A = len(A)
    cols_A = len(A[0])
    cols_B = len(B[0])
    result = [[0.0 for _ in range(cols_B)] for _ in range(rows_A)]

    for i in range(rows_A):
        for j in range(cols_B):
            result[i][j] = sum(A[i][k] * B[k][j] for k in range(cols_A))
    
    return result

def sign(a):
    if abs(a) == 0:
        return 0
    elif a.real < 0:
        return -1
    else:
        return 1
    
def normal(vec):
    return sum(abs(x)**2 for x in vec)**0.5
      
def householder_mat(A, num_i):
    n = len(A)
    V = [0.0] * n
    a = [row[num_i] for row in A]
    
    norm_x = normal(a[num_i:])
    V[num_i] = a[num_i] + sign(a[num_i]) * norm_x
    for i in range(num_i + 1, n):
        V[i] = a[i]
        
    # Нормируем v
    norm_v = normal(V)
    if norm_v == 0:
        return [[1 if r == c else 0 for c in range(n)] for r in range(n)]

    # Нормализуем v
    V = [vi / norm_v for vi in V]
    
    # Создаем единичную матрицу E
    E = [[1 if r == c else 0 for c in range(n)] for r in range(n)]
    
    # V * V^T
    V_V_T = [[V[i] * V[j] for j in range(n)] for i in range(n)]
    
    # V^T * V 
    V_T_V = sum(vi ** 2 for vi in V)
    
    # H = E - 2 * (V * V^T) / (V^T * V)
    H = [[E[i][j] - 2 * V_V_T[i][j] / V_T_V for j in range(n)] for i in range(n)]
    
    return H
    
    
def QR_decomposition(A):
    # A = Q * R
    n = len(A)
    Q = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    R = [row[:] for row in A]
    H_list = []
    
    for i in range(n - 1):
        H = householder_mat(R, i)
        H_list.append(H)
        Q = matrix_multiply(Q, H)
        R = matrix_multiply(H, R)
    return Q, R, H_list

def get_roots(A, i):
    # Корни системы двух уравнений (i и i+1) из матрицы A
    n = len(A)
    a11 = A[i][i]
    a12 = A[i][i + 1] if i + 1 < n else 0
    a21 = A[i + 1][i] if i + 1 < n else 0
    a22 = A[i + 1][i + 1] if i + 1 < n else 0
    
    # Характеристическое уравнение: det(A - λI) = 0
    # Это уравнение вида: λ^2 - (a11 + a22) * λ + (a11 * a22 - a12 * a21) = 0
    
    # Коэффициенты квадратного уравнения
    a = 1
    b = -(a11 + a22)
    c = a11 * a22 - a12 * a21
    
    # Дискриминант
    D = b * b - 4 * a * c
    if D >= 0:
        # Два вещественных корня
        root1 = (-b + cmath.sqrt(D)) / (2 * a)
        root2 = (-b - cmath.sqrt(D)) / (2 * a)
    else:
        # Комплексные корни
        real_part = -b / (2 * a)
        imag_part = cmath.sqrt(-D) / (2 * a)
        root1 = complex(real_part, imag_part)
        root2 = complex(real_part, -imag_part)
        
    # Если мнимая часть равна 0, возвращаем вещественное число
    if root1.imag == 0:
        root1 = root1.real
    if root2.imag == 0:
        root2 = root2.real

    return root1, root2

def is_complex(A, i, eps):
    Q, R, _ = QR_decomposition(A)
    A_next = matrix_multiply(R, Q)
    lambda1_1, lambda1_2 = get_roots(A, i)
    lambda2_1, lambda2_2 = get_roots(A_next, i)
    return abs(lambda1_1 - lambda2_1) <= eps and abs(lambda1_2 - lambda2_2) <= eps


def get_eigenval(A, eps, num_i, iter):
    #i-е собственное значение матрицы A
    A_i = [row[:] for row in A]
    A_92 = [row[:] for row in A]
    while True:
        Q, R, _ = QR_decomposition(A_i)
        A_i = matrix_multiply(R, Q)
        iter += 1
        if iter == 92:
            A_92 = A_i
                    
        if normal([A_i[j][num_i] for j in range(num_i + 1, len(A_i))]) <= eps:
            return A_i[num_i][num_i], A_i, iter, A_92
        elif normal([A_i[j][num_i] for j in range(num_i + 2, len(A_i))]) <= eps and is_complex(A_i, num_i, eps):
            return get_roots(A_i, num_i), A_i, iter, A_92
        

def QR_algoritmh(A, eps):
    n = len(A)
    A_i = [row[:] for row in A]
    eigen_values = []

    i = 0
    iter = 0
    while i < n:
        cur_eigen_values, A_i_plus_1, iter, A_92 = get_eigenval(A_i, eps, i, iter)
        if isinstance(cur_eigen_values, tuple):  # Комплексные собственные значения
            eigen_values.extend(cur_eigen_values)
            i += 2
        else:  # Реальные собственные значения
            eigen_values.append(cur_eigen_values)
            i += 1
                    
        A_i = A_i_plus_1
        
    return eigen_values, iter, A_i, A_92
    

def main():
    A, eps = read_file("input.txt")

    Q, R, H_list =  QR_decomposition(A)
    
    eigenval, iter, A_i, A_92= QR_algoritmh(A, eps)
    
    write_in_file("output.txt", A, eps, Q, R, H_list, eigenval, iter, A_i, A_92)

    
if __name__ == "__main__":
    main()