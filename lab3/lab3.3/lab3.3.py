import matplotlib.pyplot as plt
import math

def read_file(filename):
    with open(filename, "r") as file:
        x = list(map(float, file.readline().split()))
        y = list(map(float, file.readline().split()))
    return x, y

def write_results(filename, ls1, sum_squard_err1, ls2, sum_squard_err2):
    with open(filename, "w", encoding='utf-8') as file:
        file.write(f"Приближающий многочлен 1-ой степени\n")
        file.write(f"P(x) = {ls1[0]:.6f} + {ls1[1]:.6f}x\n")
        file.write(f"Сумма квадратов ошибок: {sum_squard_err1}\n\n")

        file.write(f"Приближающий многочлен 2-ой степени\n")
        file.write(f"P(x) = {ls2[0]:.6f} + {ls2[1]:.6f}x + {ls2[2]:.6f}x^2\n")
        file.write(f"Сумма квадратов ошибок: {sum_squard_err2}")

def plot_graphics(x, y, ls1, ls2):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'or', label='Исходные точки')

    y1 = [ls1[0] + ls1[1] * xi for xi in x]
    y2 = [ls2[0] + ls2[1] * xi + ls2[2] * xi ** 2 for xi in x]

    plt.plot(x, y1, label='Многочлен 1-ой степени')
    plt.plot(x, y2, label='Многочлен 2-ой степени')
    plt.legend()
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Аппроксимация методом наименьших квадратов')
    plt.show()

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


def solve_slay(A, b):
    
    L, U, P = lu_decomposition(A)
    
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
def least_squares_1st(x, y):
    # Система уравнений для многочлена 1-й степени:
    # n*a0 + sum_x*a1 = sum_y
    # sum_x*a0 + sum_x2*a1 = sum_xy

    n = len(x)
    sum_x = sum(x)
    sum_x2 = sum(xi ** 2 for xi in x)
    sum_y = sum(y)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))

    A = [
        [n, sum_x],
        [sum_x, sum_x2]
    ]

    b = [sum_y, sum_xy]

    coeff = solve_slay(A, b)
    return coeff

def calculate_errors(x, y, ls, degree):
    if degree == 1:
        errors = [(ls[0] + ls[1] * xi - yi) ** 2 for xi, yi in zip(x, y)]
    if degree == 2:
        errors = [(ls[0] + ls[1] * xi + ls[2] * xi ** 2 - yi) ** 2 for xi, yi in zip(x, y)]
    return sum(errors)

def least_squares_2st(x, y):
    # Система уравнений для многочлена 2-й степени:
    # n*a0 + sum_x*a1 + sum_x2*a2 = sum_y
    # sum_x*a0 + sum_x2*a1 + sum_x3*a2 = sum_xy
    # sum_x2*a0 + sum_x3*a1 + sum_x4*a2 = sum_x2y

    n = len(x)
    sum_x = sum(x)
    sum_x2 = sum(xi ** 2 for xi in x)
    sum_x3 = sum(xi ** 3 for xi in x)
    sum_x4 = sum(xi ** 4 for xi in x)
    sum_y = sum(y)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))
    sum_x2y = sum(xi ** 2 * yi for xi, yi in zip(x, y))

    A = [
        [n, sum_x, sum_x2],
        [sum_x, sum_x2, sum_x3],
        [sum_x2, sum_x3, sum_x4]
    ]

    b = [sum_y, sum_xy, sum_x2y]

    coeff = solve_slay(A, b)
    return coeff

def main():
    x, y = read_file("input.txt")

    ls1 = least_squares_1st(x, y)
    sum_squard_err1 = calculate_errors(x, y, ls1, 1)

    ls2 = least_squares_2st(x, y)
    sum_squard_err2 = calculate_errors(x, y, ls2, 2)

    write_results("output.txt", ls1, sum_squard_err1, ls2, sum_squard_err2)
    plot_graphics(x, y, ls1, ls2)

if __name__ == "__main__":
    main()