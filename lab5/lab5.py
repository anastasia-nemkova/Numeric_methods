import matplotlib.pyplot as plt
import math
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Параметры
L = 1.0
T = 1.0
a = 0.1

def U(x, t, a):
    return x + math.exp(-(math.pi ** 2) * a * t) * math.sin(math.pi * x)

def init_cond(x):
    return x + math.sin(math.pi * x)

def explicit_scheme(N, K):
    h = L / N
    tao = T / K
    sigma = a * tao / (h * h)

    if sigma > 0.5:
        print(f"σ = {sigma:.3f} > 0.5, условие устойчивости не выполнено")

    u = [[0.0 for _ in range(N + 1)] for _ in range(K + 1)]

    for i in range(N + 1):
        u[0][i] = init_cond(i * h)

    for k in range(K + 1):
        u[k][0] = 0.0
        u[k][N] = 1.0

    for k in range(0, K):
        for j in range(1, N):
            u[k + 1][j] = sigma * u[k][j + 1] + (1 - 2 * sigma) * u[k][j] + sigma * u[k][j - 1]

    return u, h, tao

def tridiagonal_matrix_algorithm(A, b):
    n = len(b)
    # Прогоночные коэффициенты
    alpha = [0.0] * (n)
    beta = [0.0] * (n)
    
    # Прямой ход
    alpha[0] = -A[0][1] / A[0][0]
    beta[0] = b[0] / A[0][0]
    
    for i in range(1, n - 1):
        denominator = A[i][i] + A[i][i-1] * alpha[i-1]
        alpha[i] = -A[i][i+1] / denominator
        beta[i] = (b[i] - A[i][i-1] * beta[i-1]) / denominator
    
    # Обратный ход
    x = [0.0] * n
    denominator = A[n-1][n-1] + A[n-1][n-2] * alpha[n-2]
    x[n-1] = (b[n-1] - A[n-1][n-2] * beta[n-2]) / denominator
    
    for i in range(n-2, -1, -1):
        x[i] = alpha[i] * x[i+1] + beta[i]
    
    return x

def implicit_scheme(N, K):
    h = L / N
    tao = T / K
    sigma = a * tao / (h * h)

    u = [[0.0 for _ in range(N + 1)] for _ in range(K + 1)]

    for i in range(N + 1):
        u[0][i] = init_cond(i * h)

    for k in range(K + 1):
        u[k][0] = 0.0
        u[k][N] = 1.0

    n = N - 1   
    for k in range(0, K):
        A = [[0.0 for _ in range(n)] for _ in range(n)]
        b = [0.0 for _ in range(n)]

        for i in range(n):
            if i == 0:
                A[i][i] = 1 + 2 * sigma
                A[i][i + 1] = -sigma
                b[i] = u[k][i + 1] + sigma * u[k + 1][0]
            elif i == n - 1:
                A[i][i - 1] = -sigma
                A[i][i] = 1 + 2 * sigma
                b[i] = u[k][i + 1] + sigma * u[k + 1][N]
            else:
                A[i][i - 1] = -sigma
                A[i][i] = 1 + 2 * sigma
                A[i][i + 1] = -sigma
                b[i] = u[k][i + 1]
                
        solve = tridiagonal_matrix_algorithm(A, b)

        for i in range(n):
            u[k + 1][i + 1] = solve[i]

    return u, h, tao

def crank_nicolson_scheme(N, K):
    h = L / N
    tao = T / K
    sigma = a * tao / (h * h)
    theta = 0.5

    u = [[0.0 for _ in range(N + 1)] for _ in range(K + 1)]

    for i in range(N + 1):
        u[0][i] = init_cond(i * h)

    for k in range(K + 1):
        u[k][0] = 0.0
        u[k][N] = 1.0
    
    n = N - 1
    for k in range(0, K):
        A = [[0.0 for _ in range(n)] for _ in range (n)]
        b = [0.0 for _ in range(n)]

        for i in range(n):
            j = i + 1

            if i == 0:
                A[i][i] = 1 + 2 * theta * sigma
                A[i][i + 1] = -theta * sigma
                b[i] = theta * sigma * u[k + 1][0] + \
                       (1 - theta) * sigma * u[k][j + 1] + \
                       (1 - 2 * (1 - theta) * sigma) * u[k][j] + \
                       (1 - theta) * sigma * u[k][j - 1]
            elif i == n - 1:
                A[i][i - 1] = -theta * sigma
                A[i][i] = 1 + 2 * theta * sigma
                b[i] = theta * sigma * u[k + 1][N] + \
                       (1 - theta) * sigma * u[k][j + 1] + \
                       (1 - 2 * (1 - theta) * sigma) * u[k][j] + \
                       (1 - theta) * sigma * u[k][j - 1]
            else:
                A[i][i - 1] = -theta * sigma
                A[i][i] = 1 + 2 * theta * sigma
                A[i][i + 1] = -theta * sigma
                b[i] = (1 - theta) * sigma * u[k][j + 1] + \
                       (1 - 2 * (1 - theta) * sigma) * u[k][j] + \
                       (1 - theta) * sigma * u[k][j - 1]
                
        solve = tridiagonal_matrix_algorithm(A, b)

        for i in range(n):
            u[k + 1][i + 1] = solve[i]

    return u, h, tao

def calculate_error(u_num, K, h, a):
    """Вычисление максимальной погрешности в последний момент времени"""
    N = len(u_num[0]) - 1
    t_final = T
    
    error = 0.0
    for i in range(N + 1):
        x = i * h
        exact = U(x, t_final, a)
        error = max(error, abs(u_num[K][i] - exact))
    
    return error

def plot_solution(u_num, method_name, h, tao, a):
    """Построение графика численного решения"""
    N = len(u_num[0]) - 1
    K = len(u_num) - 1
    
    x = [i * h for i in range(N + 1)]

    time_indices = [0, K//4, K//2, 3*K//4, K]
    
    plt.figure(figsize=(10, 6))
    for idx in time_indices:
        t = idx * tao
        y_num = u_num[idx]
        y_exact = [U(xi, t, a) for xi in x]
        
        plt.plot(x, y_num, '--', linewidth=2, label=f'{method_name}, t={t:.2f}')
        plt.plot(x, y_exact, '-', linewidth=1, alpha=0.7, label=f'Точное, t={t:.2f}')
    
    plt.xlabel('x')
    plt.ylabel('U(x,t)')
    plt.title(f'Сравнение решений: {method_name}')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_3d_solution(u_num, method_name, h, tao):
    """Построение 3D графика решения"""
    N = len(u_num[0]) - 1
    K = len(u_num) - 1
    
    x = np.array([i * h for i in range(N + 1)])
    t = np.array([k * tao for k in range(K + 1)])
    
    X, T = np.meshgrid(x, t)
    U_num = np.array(u_num)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(X, T, U_num, cmap='viridis', alpha=0.8)
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('U(x,t)')
    ax.set_title(f'Численное решение: {method_name}')
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.show()


def plot_error_evolution_all_schemes(u_exp, u_imp, u_cn, h, tao, a):
    N = len(u_exp[0]) - 1
    K = len(u_exp) - 1
    
    time_steps = list(range(0, K + 1, max(1, K // 20)))
    times = [t * tao for t in time_steps]

    errors_exp = []
    errors_imp = []
    errors_cn = []
    
    for k in time_steps:
        max_error_exp = 0.0
        max_error_imp = 0.0
        max_error_cn = 0.0
        
        for i in range(N + 1):
            x = i * h
            t_val = k * tao
            exact = U(x, t_val, a)
            
            max_error_exp = max(max_error_exp, abs(u_exp[k][i] - exact))
            max_error_imp = max(max_error_imp, abs(u_imp[k][i] - exact))
            max_error_cn = max(max_error_cn, abs(u_cn[k][i] - exact))
        
        errors_exp.append(max_error_exp)
        errors_imp.append(max_error_imp)
        errors_cn.append(max_error_cn)

    plt.figure(figsize=(12, 8))
    plt.plot(times, errors_exp, 'ro-', linewidth=2, markersize=6, label='Явная схема')
    plt.plot(times, errors_imp, 'bs-', linewidth=2, markersize=6, label='Неявная схема')
    plt.plot(times, errors_cn, 'g^-', linewidth=2, markersize=6, label='Кранк-Николсон')
    
    plt.xlabel('Время t', fontsize=12)
    plt.ylabel('Максимальная погрешность', fontsize=12)
    plt.title('Зависимость погрешности от времени для всех схем', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_error_vs_h():
    """График зависимости погрешности от шага по пространству h"""
    N_values = [5, 10, 15, 20, 25]
    K_fixed = 150
    
    errors_exp = []
    errors_imp = []
    errors_cn = []
    h_values = []
    
    for N in N_values:
        h = L / N
        h_values.append(h)
        
        u_exp, h_exp, tao_exp = explicit_scheme(N, K_fixed)
        u_imp, h_imp, tao_imp = implicit_scheme(N, K_fixed)
        u_cn, h_cn, tao_cn = crank_nicolson_scheme(N, K_fixed)
        
        errors_exp.append(calculate_error(u_exp, K_fixed, h, a))
        errors_imp.append(calculate_error(u_imp, K_fixed, h, a))
        errors_cn.append(calculate_error(u_cn, K_fixed, h, a))
    
    plt.figure(figsize=(10, 6))
    plt.plot(h_values, errors_exp, 'o-', label='Явная схема', linewidth=2)
    plt.plot(h_values, errors_imp, 's-', label='Неявная схема', linewidth=2)
    plt.plot(h_values, errors_cn, '^-', label='Кранк-Николсон', linewidth=2)

    h_ref = np.array(h_values)
    plt.plot(h_ref, h_ref**2, '--', label='O(h²)', alpha=0.7)
    
    plt.xlabel('Шаг по пространству h')
    plt.ylabel('Максимальная погрешность')
    plt.title('Зависимость погрешности от шага по пространству')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_error_vs_tau():
    """График зависимости погрешности от шага по времени τ"""
    N_fixed = 15
    K_values = [50, 100, 150, 200, 250]
    
    errors_exp = []
    errors_imp = []
    errors_cn = []
    tau_values = []
    
    for K in K_values:
        tau = T / K
        tau_values.append(tau)
        
        u_exp, h_exp, tao_exp = explicit_scheme(N_fixed, K)
        u_imp, h_imp, tao_imp = implicit_scheme(N_fixed, K)
        u_cn, h_cn, tao_cn = crank_nicolson_scheme(N_fixed, K)
        
        errors_exp.append(calculate_error(u_exp, K, h_exp, a))
        errors_imp.append(calculate_error(u_imp, K, h_imp, a))
        errors_cn.append(calculate_error(u_cn, K, h_cn, a))
    
    plt.figure(figsize=(10, 6))
    plt.plot(tau_values, errors_exp, 'o-', label='Явная схема', linewidth=2)
    plt.plot(tau_values, errors_imp, 's-', label='Неявная схема', linewidth=2)
    plt.plot(tau_values, errors_cn, '^-', label='Кранк-Николсон', linewidth=2)
    
    tau_ref = np.array(tau_values)
    plt.plot(tau_ref, tau_ref, '--', label='O(τ)', alpha=0.7)
    plt.plot(tau_ref, tau_ref**2, '--', label='O(τ²)', alpha=0.7)
    
    plt.xlabel('Шаг по времени τ')
    plt.ylabel('Максимальная погрешность')
    plt.title('Зависимость погрешности от шага по времени')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    N = 20
    K = 200 
    
    print("Решение начально-краевой задачи для уравнения теплопроводности")
    print(f"Параметры: L={L}, T={T}, a={a}")
    print(f"Сетка: N={N}, K={K}")
    
    # Вычисление решений
    u_exp, h_exp, tao_exp = explicit_scheme(N, K)
    u_imp, h_imp, tao_imp = implicit_scheme(N, K)
    u_cn, h_cn, tao_cn = crank_nicolson_scheme(N, K)

    # Построение графиков решений
    plot_solution(u_exp, "Явная схема", h_exp, tao_exp, a)
    plot_solution(u_imp, "Неявная схема", h_imp, tao_imp, a)
    plot_solution(u_cn, "Кранк-Николсон", h_cn, tao_cn, a)
    
    # 3D графики решений
    # plot_3d_solution(u_exp, "Явная схема", h_exp, tao_exp)
    # plot_3d_solution(u_imp, "Неявная схема", h_imp, tao_imp)
    # plot_3d_solution(u_cn, "Кранк-Николсон", h_cn, tao_cn)

    plot_error_evolution_all_schemes(u_exp, u_imp, u_cn, h_exp, tao_exp, a)
    
    # Графики зависимости от сеточных параметров
    plot_error_vs_h()
    plot_error_vs_tau()
    
    # Таблица сравнения методов
    print("\nСравнение методов:")
    print("N\tK\th\t\tτ\t\tσ\t\tПогрешность (явная)\tПогрешность (неявная)\tПогрешность (КН)")
    
    N_values = [10, 20, 40, 80]
    K_values = [50, 100, 400, 1600]
    
    for N_test, K_test in zip(N_values, K_values):
        u_exp, h_test, tao_test = explicit_scheme(N_test, K_test)
        u_imp, _, _ = implicit_scheme(N_test, K_test)
        u_cn, _, _ = crank_nicolson_scheme(N_test, K_test)
        
        error_exp = calculate_error(u_exp, K_test, h_test, a)
        error_imp = calculate_error(u_imp, K_test, h_test, a)
        error_cn = calculate_error(u_cn, K_test, h_test, a)
        
        sigma_test = a * tao_test / (h_test * h_test)
        
        print(f"{N_test}\t{K_test}\t{h_test:.6f}\t{tao_test:.6f}\t{sigma_test:.4f}\t"
              f"{error_exp:.8f}\t\t{error_imp:.8f}\t\t{error_cn:.8f}")

if __name__ == "__main__":
    main()