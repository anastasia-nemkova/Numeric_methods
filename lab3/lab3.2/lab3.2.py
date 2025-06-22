import math
import numpy as np
import matplotlib.pyplot as plt

def read_file(filename):
    with open(filename, "r") as file:
        x_test = float(file.readline())
        x = list(map(float, file.readline().split()))
        y = list(map(float, file.readline().split()))
    return x_test, x, y

def s(x_val, coeffs, x_i):
    """Вычисление значения сплайна в точке x_val на отрезке [x_i, x_i+1]"""
    a, b, c, d = coeffs
    dx = x_val - x_i
    return a + b * dx + c * dx**2 + d * dx**3

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
        
    # Обратный ход: вычисляем x
    x = [0.0 for _ in range(n)]
    x[n - 1] = (b[n - 1] - A[n - 1][n - 2] * Q[n - 2]) / (A[n - 1][n - 2] * P[n - 2] + A[n - 1][n - 1])
    for i in range(n - 1, 0, -1):
         x[i - 1] = P[i - 1] * x[i] + Q[i - 1]
         
    return x

def build_spline(x, y):
    n = len(x) - 1
    
    # Шаги h_i
    h = [x[i + 1] - x[i] for i in range(n)]
    
    # Построение трехдиагональной матрицы для нахождения c_i
    A = [[0]*(n + 1) for _ in range(n + 1)]
    b = [0]*(n + 1)
    
    # Естественные граничные условия (c_0 = c_n = 0)
    A[0][0] = 1
    A[n][n] = 1
    
    for i in range(1, n):
        A[i][i - 1] = h[i - 1]
        A[i][i] = 2 * (h[i - 1] + h[i])
        A[i][i + 1] = h[i]
        b[i] = 3 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
    
    # Решаем систему для c_i
    c = tridiagonal_matrix_algorithm(A, b)

    
    # Вычисляем коэффициенты a, b, d

    # a = f(x_i)
    a = [y[i] for i in range(n)]

    # b = (a_i - a_i-1) / h_i - (2c_i + c_i-1) * h_i / 3
    b = []

    #d_i = (c_i+1 - c_i) / 3h_i
    d = []
    for i in range(n):
        b.append((y[i + 1] - y[i]) / h[i] - h[i] * (2 * c[i] + c[i + 1]) / 3)
        d.append((c[i + 1] - c[i]) / (3 * h[i]))
    
    # Собираем коэффициенты для каждого отрезка
    coeffs = []
    for i in range(n):
        coeffs.append((a[i], b[i], c[i], d[i]))
    
    return coeffs

def find_interval(x, x_val):
    """Находит индекс интервала, в который попадает x_val"""
    for i in range(len(x) - 1):
        if x[i] <= x_val <= x[i + 1]:
            return i
    return -1

def write_results(filename, x_test, y_test, x, coeffs):
    with open(filename, "w", encoding='utf-8') as file:
        file.write(f"Значение сплайна в точке x* = {x_test}: {y_test}\n\n")
        file.write("Уравнения сплайна для каждого промежутка:\n")
        for i in range(len(coeffs)):
            a, b, c, d = coeffs[i]
            file.write(f"S_{i}(x) = {a:.4f} + {b:.4f}(x - {x[i]:.4f}) + {c:.4f}(x - {x[i]:.4f})² + {d:.4f}(x - {x[i]:.4f})³, x ∈ [{x[i]:.4f}, {x[i+1]:.4f}]\n")

def plot_spline(x, y, coeffs, x_test, y_test):
    plt.figure(figsize=(10, 6))
    
    # Отображаем исходные точки
    plt.plot(x, y, 'o', label='Узлы интерполяции')
    
    # Отображаем точку, в которой вычисляли значение
    plt.plot(x_test, y_test, 'ro', label=f'Точка x* = {x_test:.2f}')
    
    # Строим сплайн
    x_vals = np.linspace(min(x), max(x), 1000)
    y_vals = []
    for x_val in x_vals:
        i = find_interval(x, x_val)
        if i >= 0:
            y_vals.append(s(x_val, coeffs[i], x[i]))
        else:
            y_vals.append(None)
    plt.plot(x_vals, y_vals, 'b-', label='Кубический сплайн')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Кубический сплайн с естественными граничными условиями')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    x_test, x, y = read_file("input.txt")
    coeffs = build_spline(x, y)
    
    i = find_interval(x, x_test)
    if i >= 0:
        y_test = s(x_test, coeffs[i], x[i])
    else:
        y_test = None
    
    write_results("output.txt", x_test, y_test, x, coeffs)
    
    plot_spline(x, y, coeffs, x_test, y_test)

if __name__ == "__main__":
    main()