def read_from_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    n = len(lines) - 1
    A = [[0.0] * n for _ in range(n)]
    
    A[0][0], A[0][1] = map(float, lines[0].split())
    
    for i in range(1, n - 1):
        A[i][i - 1], A[i][i], A[i][i + 1] = map(float, lines[i].split())
    
    A[n - 1][n - 2], A[n - 1][n - 1] = map(float, lines[n - 1].split())
    
    b = list(map(float, lines[-1].split()))
    return A, b   

def write_in_file(filename, A, b, x):
    with open(filename, 'w') as file:
        file.write("Matrix A:\n")
        for row in A:
            file.write(" ".join(f"{val:.2f}" for val in row) + "\n")
        
        file.write("\nVector b:\n")
        file.write(" ".join(f"{val:.2f}" for val in b) + "\n")
        
        file.write("\nSolution x:\n")
        file.write(" ".join(f"{val:.2f}" for val in x) + "\n")
    checks(filename, x, A, b)

def tridiagonal_matrix_algorithm(A, b):
    n = len(A)
    P = [0.0 for _ in range(n)]
    Q = [0.0 for _ in range(n)]
    P[0] = -A[0][1] / A[0][0]
    Q[0] = b[0] / A[0][0]
    
    # Прямой ход: вычисляем прогоночные коэффициенты P, Q
    for i in range(1, n - 1):
        P[i] = -A[i][i + 1] / (A[i][i - 1] * P[i - 1] + A[i][i])
        Q[i] = (b[i] - A[i][i - 1] * Q[i - 1]) / (A[i][i - 1] * P[i - 1] + A[i][i])
        
    # Обратный ход: вычисляем х
    x = [0.0 for _ in range(n)]
    x[n - 1] = (b[n - 1] - A[n - 1][n - 2] * Q[n - 2]) / (A[n - 1][n - 2] * P[n - 2] + A[n - 1][n - 1])
    for i in range(n - 1, 0, -1):
         x[i - 1] = P[i - 1] * x[i] + Q[i - 1]
         
    return x
    
def checks(filename, x, A, b):
    with open(filename, "a") as file:
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

def main():
    A, b = read_from_file("input.txt")
    x = tridiagonal_matrix_algorithm(A, b)
    write_in_file("output.txt", A, b, x)
 
if __name__ == "__main__":
    main()