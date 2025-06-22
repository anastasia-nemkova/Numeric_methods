import math

def read_file(filename):
    with open(filename, "r") as file:
        lines = file.readlines()
    A = []
    for line in lines[:-2]:
        A.append(list(map(float, line.split())))
    
    b = list(map(float, lines[-2].split()))
    eps = float(lines[-1])
    return A, b, eps

def write_in_file(filename, A, b,  eps, x, iterations, x_s, iterations_s):
    with open(filename, "w") as file:
        file.write("Matrix A:\n")
        for row in A:
            file.write(" ".join(f"{val:.2f}" for val in row) + "\n")
        
        file.write("\nVector b:\n")
        file.write(" ".join(f"{val:.2f}" for val in b) + "\n")
        
        file.write(f"\nEps: {eps}\n")
        
        file.write("\nSimple iteration method:\n")
        file.write("\nSolution x:\n")
        file.write(" ".join(f"{val:.2f}" for val in x) + "\n")
        file.write(f"Iterations: {iterations}\n")
        
        file.write("\nSeidel method:\n")
        file.write("\nSolution x:\n")
        file.write(" ".join(f"{val:.2f}" for val in x_s) + "\n")
        file.write(f"Iterations: {iterations_s}\n")
        
        file.write("\nChecks:\n")
        
        
        # Verify that A * x = b
        Ax = [sum(A[i][j] * x[j] for j in range(len(A[i]))) for i in range(len(A))]
        file.write("\nResult of A * x:\n")
        for value in Ax:
            file.write(f"{value:.2f} ")
        file.write("\nOriginal b:\n")
        for value in b:
            file.write(f"{value} ")
        file.write("\n")

def normal(x):
    n = len(x)
    norm = 0
    
    for i in range(n):
        norm += x[i] * x[i]
    return math.sqrt(norm)

def normal_mat(alf):
    n = len(alf)
    norm = 0
    
    for i in range(n):
        for j in range(n):
            norm += alf[i][j] * alf[i][j]
            
    return math.sqrt(norm)

def vector_diff(x1, x2):
    return [x1[i] - x2[i] for i in range(len(x1))]

def finish_iter(x_new, x, alpha, eps):
    diff = vector_diff(x_new, x)
    norm_x = normal(diff)
    
    if normal_mat(alpha) >= 1:
        return norm_x <= eps
    else:
        coeff = normal_mat(alpha) / (1 - normal_mat(alpha))
        return coeff * norm_x <= eps

def simple_iteration_method(A, b, eps, max_iter=100):
    n = len(A)
    alpha = [[0.0 for _ in range(n)] for _ in range(n)]
    beta = [0.0 for _ in range(n)]
    
    for i in range(n):
        beta[i] = b[i] / A[i][i]
        for j in range(n):
            if i == j:
                alpha[i][j] = 0.0
            else:
                alpha[i][j] = -A[i][j] / A[i][i]
                
    x = [0.0] * n # начальное приближение(любое)
    iter = 0
    while iter < max_iter:
        x_new = [0.0] * n
        for i in range(n):
            temp_sum = 0
            for j in range(n):
                temp_sum += alpha[i][j] * x[j]
            x_new[i] = beta[i] + temp_sum
        
        if finish_iter(x_new, x, alpha, eps):
            break
        
        x = x_new
        iter += 1
    return x_new, iter

def matrix_diff(A, B):
    rows_A = len(A)
    cols_A = len(A[0])
    result = [[0.0 for _ in range(cols_A)] for _ in range(rows_A)]
    
    for i in range(rows_A):
        for j in range(cols_A):
            result[i][j] = A[i][j] - B[i][j]
    
    return result

def pivotize(A):
    n = len(A)
    P = [[float(i == j) for i in range(n)] for j in range(n)]
    
    for i in range(n):
        max_elem = max(range(i, n), key=lambda j: abs(A[j][i]))
        if i != max_elem:
            P[i], P[max_elem] = P[max_elem], P[i]
    return P

def matrix_multiply(A, B):
    rows_A = len(A)
    cols_A = len(A[0])
    cols_B = len(B[0])
    result = [[0.0 for _ in range(cols_B)] for _ in range(rows_A)]

    for i in range(rows_A):
        for j in range(cols_B):
            result[i][j] = sum(A[i][k] * B[k][j] for k in range(cols_A))
    
    return result

def lu_decomposition(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]
    P = pivotize(A)
    PA = matrix_multiply(P, A)
    
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

def vector_multiply_matrix(mat, vec):
    result = []
    
    for row in mat:
        row_result = sum(row[i] * vec[i] for i in range(len(row)))
        result.append(row_result)
    
    return result

def vector_add(v1, v2):
    return [v1[i] + v2[i] for i in range(len(v1))]

def finish_seidel(x_new, x, C, alpha, eps):
    diff = vector_diff(x_new, x)
    norm_x = normal(diff)
    
    if normal_mat(alpha) >= 1:
        return norm_x <= eps
    else:
        coeff = normal_mat(C) / (1 - normal_mat(alpha))
        return coeff * norm_x <= eps

def seidel(A, b, eps, max_iter=100):
    n = len(A)
    alpha = [[0.0 for _ in range(n)] for _ in range(n)]
    beta = [0.0 for _ in range(n)]
    
    for i in range(n):
        beta[i] = b[i] / A[i][i]
        for j in range(n):
            if i == j:
                alpha[i][j] = 0.0
            else:
                alpha[i][j] = -A[i][j] / A[i][i]
                
    B = [[0.0 for _ in range(n)] for _ in range(n)]
    C = [[0.0 for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        for j in range(i):
            B[i][j] = alpha[i][j]
        for j in range(i, n):
            C[i][j] = alpha[i][j]
    
    E = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    diffEB = matrix_diff(E, B)
    inv_EB = inverse_matrix(diffEB)
    tmp1 = matrix_multiply(inv_EB, C)
    tmp2 = vector_multiply_matrix(inv_EB, beta)
    
    x = tmp2
    iter = 0
    while iter < max_iter:
        x_new = vector_add(tmp2, vector_multiply_matrix(tmp1, x))
        
        if finish_seidel(x_new, x, C, alpha, eps):
            break
        
        x = x_new
        iter += 1
    return x, iter
       
def main():
    A, b, eps = read_file("input.txt")
    
    x , iterations = simple_iteration_method(A, b, eps)
    
    x_s, iterations_s = seidel(A, b, eps)
    write_in_file("output.txt", A, b, eps, x, iterations, x_s, iterations_s)
    

if __name__ == "__main__":
    main()