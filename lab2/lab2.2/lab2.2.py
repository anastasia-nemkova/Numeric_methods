import math
import matplotlib.pyplot as plt
import numpy as np

# f1 и f2 — функции исходной системы
def f1(x):
    return x[0] - math.cos(x[1]) - 3

def f2(x):
    return x[1] - math.sin(x[0]) - 3

# Производные f1 и f2
def df1_dx1(x):
    return 1

def df1_dx2(x):
    return math.sin(x[1])

def df2_dx1(x):
    return -math.cos(x[0])

def df2_dx2(x):
    return 1

# Итерационные функции φ1 и φ2
def fi1(x):
    # x1 = fi(x2)
    return math.cos(x[1]) + 3

def fi2(x):
    #x2 = fi(x1)
    return math.sin(x[0]) + 3

def dfi1_dx1(x):
    # ∂φ1/∂x1 = 0
    return 0

def dfi1_dx2(x):
    # ∂φ1/∂x2 = -sin(x2)
    return -math.sin(x[1])

def dfi2_dx1(x):
    # ∂φ2/∂x1 = cos(x1)
    return math.cos(x[0])

def dfi2_dx2(x):
    # ∂φ2/∂x2 = 0
    return 0

def jacobian(x):
    return [
        [df1_dx1(x), df1_dx2(x)],
        [df2_dx1(x), df2_dx2(x)]
    ]
    
def F(x):
    return [f1(x), f2(x)]

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

   
def get_q(x):
    '''
    q = max|| Jφ|| 
    '''
    max_fi1 = abs(dfi1_dx1(x)) + abs(dfi1_dx2(x))
    max_fi2 = abs(dfi2_dx1(x)) + abs(dfi2_dx2(x))
    return max(max_fi1, max_fi2)

def L2_norm(x):
    abs_x = (abs(i) for i in x)
    return max(abs_x)

def iteration_method(x0, y0, eps, max_iter = 100):
    '''
    x_k = fi(x_(k - 1))
    eps_k = q / (1 - q) * ||x_k - x_(k - 1)||
    eps_k <= eps => finish
    '''
    x_prev= [x0, y0]
    trajectory = [x_prev[:]]
    errors = []
    
    q = get_q(x_prev)
    iter = 0
    
    while iter < max_iter:
        iter += 1
        
        x = [fi1(x_prev), fi2(x_prev)] 
        trajectory.append(x[:])
        
        diff = [x[i] - x_prev[i] for i in range(len(x))]
        eps_now = q /(1 - q) * L2_norm(diff)
        errors.append(eps_now)
        
        if eps_now <= eps:
            break
        
        x_prev = x 
        
    return x, iter, trajectory, errors

def Newton_method(x0, y0, eps, max_iter=100):
    '''
    J(f(x_(k - 1))) * delta(x_k) = - f(x_(k - 1))
    eps_k = || delta(x_k) ||
    '''
    
    x_prev = [x0, y0]
    trajectory = [x_prev[:]]
    errors = []
    
    iter = 0
    while iter < max_iter:
        iter += 1
        J = jacobian(x_prev)        
        fx = F(x_prev)
        
        delta = solve_slay(J, [-val for val in fx])
        x = [x_prev[i] + delta[i] for i in range(len(x_prev))]
        trajectory.append(x[:])
        
        diff = [x[i] - x_prev[i] for i in range(len(x_prev))]
        eps_now = L2_norm(diff)
        errors.append(eps_now)
        
        if eps_now < eps:
            break
        x_prev = x 
        
    return x, iter, trajectory, errors
    
def plot_graph(x_range, y_range, iter_points, newton_points, iter_errors, newton_errors):
    x_vals = np.linspace(x_range[0], x_range[1], 400)
    y_vals = np.linspace(y_range[0], y_range[1], 400)
    X, Y = np.meshgrid(x_vals, y_vals)

    Z1 = X - np.cos(Y) - 3
    Z2 = Y - np.sin(X) - 3  

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))

    # --- 1. Cистема уравнений ---
    ax1.contour(X, Y, Z1, levels=[0], colors='green')
    ax1.contour(X, Y, Z2, levels=[0], colors='blue')
    ax1.set_title("Система уравнений")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.grid(True)
    ax1.legend(
        [plt.Line2D([0], [0], color='green'), plt.Line2D([0], [0], color='blue')],
        ['f1 = x1 - cos(x2) - 3', 'f2 = x2 - sin(x1) - 3']
    )

    # --- 2. Траектории итераций ---
    iter_x, iter_y = zip(*iter_points)
    newton_x, newton_y = zip(*newton_points)

    ax2.plot(iter_x, iter_y, marker='o', linestyle='-', color='orange', label=f"Простая итерация ({len(iter_points) - 1} ит.)")
    ax2.plot(newton_x, newton_y, marker='x', linestyle='-', color='red', label=f"Ньютон ({len(newton_points) - 1} ит.)")
    ax2.set_title("Траектории итераций")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.grid(True)
    ax2.legend()

    # --- 3. Погрешности ---
    ax3.plot(range(1, len(iter_errors) + 1), iter_errors, marker='o', color='orange', label="Простая итерация")
    ax3.plot(range(1, len(newton_errors) + 1), newton_errors, marker='x', color='red', label="Ньютон")
    ax3.set_title("Погрешность от номера итерации")
    ax3.set_xlabel("Номер итерации")
    ax3.set_ylabel("Погрешность")
    ax3.set_yscale('log') 
    ax3.grid(True, which='major', linestyle='--', linewidth=1)
    ax3.legend()

    plt.tight_layout()
    plt.show()


    
def read_file(filename):
    with open(filename, 'r') as file:
        eps = float(file.readline())
        x_range = list(map(float, file.readline().split()))
        y_range = list(map(float, file.readline().split()))
    return eps, x_range, y_range

def write_in_file(filename, eps, x0, y0, root_iter, iterations_iter, root_newton, iterations_newton):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write("Система нелинейных уравнений:\n")
        file.write("x1 - cos(x2) = 3\n")
        file.write("x2 - sin(x1) = 3\n\n")

        file.write(f"Заданная точность: ε = {eps}\n")
        file.write(f"Начальное приближение: x0 = {x0}, y0 = {y0}\n\n")

        file.write("\nМетод простой итерации:\n")
        file.write(f"Решение: x = {root_iter[0]}, y = {root_iter[1]}\n")
        file.write(f"Количество итераций: {iterations_iter}\n")
        file.write("Проверка решения:\n")
        file.write(f"f1 = {f1(root_iter)}\n")
        file.write(f"f2 = {f2(root_iter)}\n")
        
        file.write("\nМетод Ньютона:\n")
        file.write(f"Решение: x = {root_newton[0]}, y = {root_newton[1]}\n")
        file.write(f"Количество итераций: {iterations_newton}\n")
        file.write("Проверка решения:\n")
        file.write(f"f1 = {f1(root_newton)}\n")
        file.write(f"f2 = {f2(root_newton)}\n")

def main():
    eps, x_range, y_range = read_file("input.txt")

    x0 = (x_range[0] + x_range[1]) / 2
    y0 = (y_range[0] + y_range[1]) / 2
    
    x_iter, iter_iter, trajectory_iter, errors_iter = iteration_method(x0, y0, eps)
    
    x_newt, iter_newt, trajectory_newt, errors_Newt = Newton_method(x0, y0, eps)
    
    write_in_file("output.txt", eps, x0, y0, x_iter, iter_iter, x_newt, iter_newt)

    plot_graph(x_range, y_range, trajectory_iter, trajectory_newt, errors_iter, errors_Newt)
    
if __name__ == "__main__":
    main()