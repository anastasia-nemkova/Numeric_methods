import math
import matplotlib.pyplot as plt
import numpy as np

def U(x, y, t, a, mu1, mu2):
    return math.cos(mu1 * x) * math.cos(mu2 * y) * math.exp(-(mu1 * mu1 + mu2 * mu2) * a * t)

def phi(x, y, mu1, mu2):
    return math.cos(mu1 * x) * math.cos(mu2 * y)

def phi1(y, t, a, mu1, mu2):
    return math.cos(mu2 * y) * math.exp(-(mu1 * mu1 + mu2 * mu2) * a * t)

def phi2(y, t, a, mu1, mu2):
    Lx = (math.pi * mu1) / 2
    return math.cos(mu1 * Lx) * math.cos(mu2 * y) * math.exp(-(mu1 * mu1 + mu2 * mu2) * a * t)

def phi3(x, t, a, mu1, mu2):
    return math.cos(mu1 * x) * math.exp(-(mu1 * mu1 + mu2 * mu2) * a * t)

def phi4(x, t, a, mu1, mu2):
    Ly = (math.pi * mu2) / 2
    return math.cos(mu1 * x) * math.cos(mu2 * Ly) * math.exp(-(mu1 * mu1 + mu2 * mu2) * a * t)

def calc_error(u_num, u_anal, nx, ny):
    max_error = 0.0
    avg_error = 0.0
    errs = [[0.0 for _ in range(ny)] for _ in range(nx)]

    for i in range(nx):
        for j in range(ny):
            err = abs(u_num[i][j] - u_anal[i][j])
            errs[i][j] = err
            if err > max_error:
                max_error = err
            avg_error += err

    avg_error /= (nx * ny)
    return max_error, avg_error, errs

def three_diag(A, b):
    n = len(A)

    v = [0 for _ in range(n)]
    u = [0 for _ in range(n)]
    v[0] = A[0][1] / -A[0][0]
    u[0] = b[0] / A[0][0]
    for i in range(1, n-1):
        v[i] = A[i][i+1] / (-A[i][i] - A[i][i-1] * v[i-1])
        u[i] = (A[i][i-1] * u[i-1] - b[i]) / (-A[i][i] - A[i][i-1] * v[i-1])
    v[n-1] = 0
    u[n-1] = (A[n-1][n-2] * u[n-2] - b[n-1]) / (-A[n-1][n-1] - A[n-1][n-2] * v[n-2])

    x = [0 for _ in range(n)]
    x[n-1] = u[n-1]
    for i in range(n-1, 0, -1):
        x[i-1] = v[i-1] * x[i] + u[i-1]
    return x

def scheme_variable_directions(mu1, mu2, a, nx, ny, nt, T):
    Lx = (math.pi * mu1) / 2
    Ly = (math.pi * mu2) / 2
    hx = Lx / (nx - 1)
    hy = Ly / (ny - 1)
    tau = T / nt

    x = [i * hx for i in range(nx)]
    y = [j * hy for j in range(ny)]
    t = [k * tau for k in range(nt + 1)]

    u_curr = [[0.0 for _ in range(ny)] for _ in range(nx)]
    u_half = [[0.0 for _ in range(ny)] for _ in range(nx)]
    u_next = [[0.0 for _ in range(ny)] for _ in range(nx)]

    errors = []

    for i in range(nx):
        for j in range(ny):
            u_curr[i][j] = phi(x[i], y[j], mu1, mu2)

    sigma_x = a * tau / (2 * hx * hx)
    sigma_y = a * tau / (2 * hy * hy)

    for k in range(1, nt + 1):
        t_curr = t[k - 1]
        t_half = t_curr + tau / 2
        t_next = t_curr + tau

        for i in range(nx):
            u_half[i][0] = phi3(x[i], t_half, a, mu1, mu2)
            u_half[i][ny - 1] = phi4(x[i], t_half, a, mu1, mu2)
            u_next[i][0] = phi3(x[i], t_next, a, mu1, mu2)
            u_next[i][ny - 1] = phi4(x[i], t_next, a, mu1, mu2)

        for j in range(ny):
            u_half[0][j] = phi1(y[j], t_half, a, mu1, mu2)
            u_half[nx - 1][j] = phi2(y[j], t_half, a, mu1, mu2)
            u_next[0][j] = phi1(y[j], t_next, a, mu1, mu2)
            u_next[nx - 1][j] = phi2(y[j], t_next, a, mu1, mu2)

        for j in range(1, ny - 1):
            A = [[0.0 for _ in range(nx - 2)] for _ in range(nx - 2)]
            b = [0.0 for _ in range(nx - 2)]

            A[0][0] = 2 * hx * hx * hy * hy + 2 * a * tau * hy * hy
            A[0][1] = -a * tau * hy * hy
            for i in range(1, len(A) - 1):
                A[i][i - 1] = -a * tau * hy * hy
                A[i][i] = 2 * hx * hx * hy * hy + 2 * a * tau * hy * hy
                A[i][i + 1] = -a * tau * hy * hy
            A[-1][-2] = -a * tau * hy * hy
            A[-1][-1] = 2 * hx * hx * hy * hy + 2 * a * tau * hy * hy

            for i in range(1, nx - 1):
                b[i - 1] = (
                    u_curr[i][j - 1] * a * tau * hx * hx
                    + u_curr[i][j] * (2 * hx * hx * hy * hy - 2 * a * tau * hx * hx)
                    + u_curr[i][j + 1] * a * tau * hx * hx
                )
            b[0] += a * tau * hy * hy * phi1(y[j], t_half, a, mu1, mu2)
            b[-1] += a * tau * hy * hy * phi2(y[j], t_half, a, mu1, mu2)
            
            solution = three_diag(A, b)
            for i in range(1, nx - 1):
                u_half[i][j] = solution[i - 1]

        for i in range(1, nx - 1):
            A = [[0.0 for _ in range(ny - 2)] for _ in range(ny - 2)]
            b = [0.0 for _ in range(ny - 2)]

            A[0][0] = 2 * hx * hx * hy * hy + 2 * a * tau * hx * hx
            A[0][1] = -a * tau * hx * hx
            for j in range(1, len(A) - 1):
                A[j][j - 1] = -a * tau * hx * hx
                A[j][j] = 2 * hx * hx * hy * hy + 2 * a * tau * hx * hx
                A[j][j + 1] = -a * tau * hx * hx
            A[-1][-2] = -a * tau * hx * hx
            A[-1][-1] = 2 * hx * hx * hy * hy + 2 * a * tau * hx * hx

            for j in range(1, ny - 1):
                b[j - 1] = (
                    u_half[i - 1][j] * a * tau * hy * hy
                    + u_half[i][j] * (2 * hx * hx * hy * hy - 2 * a * tau * hy * hy)
                    + u_half[i + 1][j] * a * tau * hy * hy
                )
            b[0] += a * tau * hx * hx * phi3(x[i], t_next, a, mu1, mu2)
            b[-1] += a * tau * hx * hx * phi4(x[i], t_next, a, mu1, mu2)
            
            solution = three_diag(A, b)
            for j in range(1, ny - 1):
                u_next[i][j] = solution[j - 1]

        u_analytical = [[U(x[i], y[j], t_next, a, mu1, mu2) for j in range(ny)] for i in range(nx)]
        max_err, avg_err, _ = calc_error(u_next, u_analytical, nx, ny)
        errors.append((t_next, max_err, avg_err))

        for i in range(nx):
            for j in range(ny):
                u_curr[i][j] = u_next[i][j]

    return u_curr, errors

def scheme_fractional_steps(mu1, mu2, a, nx, ny, nt, T):
    Lx = (math.pi * mu1) / 2
    Ly = (math.pi * mu2) / 2
    hx = Lx / (nx - 1)
    hy = Ly / (ny - 1)
    tau = T / nt

    x = [i * hx for i in range(nx)]
    y = [j * hy for j in range(ny)]
    t = [k * tau for k in range(nt + 1)]

    u_curr = [[0.0 for _ in range(ny)] for _ in range(nx)]
    u_half = [[0.0 for _ in range(ny)] for _ in range(nx)]
    u_next = [[0.0 for _ in range(ny)] for _ in range(nx)]

    errors = []

    for i in range(nx):
        for j in range(ny):
            u_curr[i][j] = phi(x[i], y[j], mu1, mu2)

    for k in range(1, nt + 1):
        t_curr = t[k - 1]
        t_half = t_curr + tau / 2
        t_next = t_curr + tau

        for i in range(nx):
            u_half[i][0] = phi3(x[i], t_half, a, mu1, mu2)
            u_half[i][ny-1] = phi4(x[i], t_half, a, mu1, mu2)
            u_next[i][0] = phi3(x[i], t_next, a, mu1, mu2)
            u_next[i][ny-1] = phi4(x[i], t_next, a, mu1, mu2)

        for j in range(ny):
            u_half[0][j] = phi1(y[j], t_half, a, mu1, mu2)
            u_half[nx-1][j] = phi2(y[j], t_half, a, mu1, mu2)
            u_next[0][j] = phi1(y[j], t_next, a, mu1, mu2)
            u_next[nx-1][j] = phi2(y[j], t_next, a, mu1, mu2)

        for j in range(1, ny - 1):
            A = [[0.0 for _ in range(nx - 2)] for _ in range(nx - 2)]
            b = [0.0 for _ in range(nx - 2)]

            A[0][0] = hx * hx + 2 * a * tau
            A[0][1] = -a * tau
            for i in range(1, len(A) - 1):
                A[i][i - 1] = -a * tau
                A[i][i] = hx * hx + 2 * a * tau
                A[i][i + 1] = -a * tau
            A[-1][-2] = -a * tau
            A[-1][-1] = hx * hx + 2 * a * tau

            for i in range(1, nx - 1):
                b[i - 1] = u_curr[i][j] * hx * hx
            b[0] += a * tau * phi1(y[j], t_half, a, mu1, mu2)
            b[-1] += a * tau * phi2(y[j], t_half, a, mu1, mu2)
            
            solution = three_diag(A, b)
            for i in range(1, nx - 1):
                u_half[i][j] = solution[i - 1]

        for i in range(1, nx - 1):
            A = [[0.0 for _ in range(ny - 2)] for _ in range(ny - 2)]
            b = [0.0 for _ in range(ny - 2)]

            A[0][0] = hy * hy + 2 * a * tau
            A[0][1] = -a * tau
            for j in range(1, len(A) - 1):
                A[j][j - 1] = -a * tau
                A[j][j] = hy * hy + 2 * a * tau
                A[j][j + 1] = -a * tau
            A[-1][-2] = -a * tau
            A[-1][-1] = hy * hy + 2 * a * tau

            for j in range(1, ny - 1):
                b[j - 1] = u_half[i][j] * hy * hy
            b[0] += a * tau * phi3(x[i], t_next, a, mu1, mu2)
            b[-1] += a * tau * phi4(x[i], t_next, a, mu1, mu2)
            
            solution = three_diag(A, b)
            for j in range(1, ny - 1):
                u_next[i][j] = solution[j - 1]

        u_analytical = [[U(x[i], y[j], t_next, a, mu1, mu2) for j in range(ny)] for i in range(nx)]
        max_err, avg_err, _ = calc_error(u_next, u_analytical, nx, ny)
        errors.append((t_next, max_err, avg_err))

        for i in range(nx):
            for j in range(ny):
                u_curr[i][j] = u_next[i][j]

    return u_curr, errors

def plot_solutions_comparison(x, y, u_adi, u_fs, u_anal, title):
    plt.figure(figsize=(15, 10))

    y_indices = [len(y)//4, len(y)//2, 3*len(y)//4]
    y_labels = [f'y = {y[i]:.3f}' for i in y_indices]
    colors_y = ['red', 'blue', 'green']
    
    for idx, (y_idx, color, label) in enumerate(zip(y_indices, colors_y, y_labels)):
        plt.subplot(2, 3, idx + 1)
        
        u_adi_y = [u_adi[i][y_idx] for i in range(len(x))]
        u_fs_y = [u_fs[i][y_idx] for i in range(len(x))]
        u_anal_y = [u_anal[i][y_idx] for i in range(len(x))]
 
        plt.plot(x, u_anal_y, color='black', linestyle='-', linewidth=3, label='Аналитическое')
        plt.plot(x, u_adi_y, color=color, linestyle='--', linewidth=2, label='МПН')
        plt.plot(x, u_fs_y, color=color, linestyle=':', linewidth=2, label='Дробные шаги')
        
        plt.xlabel('x')
        plt.ylabel('u(x,y)')
        plt.title(f'{title}\nПри {label}')
        plt.legend()
        plt.grid(True, alpha=0.3)

    x_indices = [len(x)//4, len(x)//2, 3*len(x)//4]
    x_labels = [f'x = {x[i]:.3f}' for i in x_indices]
    colors_x = ['orange', 'purple', 'brown']
    
    for idx, (x_idx, color, label) in enumerate(zip(x_indices, colors_x, x_labels)):
        plt.subplot(2, 3, idx + 4)
        
        u_adi_x = [u_adi[x_idx][j] for j in range(len(y))]
        u_fs_x = [u_fs[x_idx][j] for j in range(len(y))]
        u_anal_x = [u_anal[x_idx][j] for j in range(len(y))]
        
        plt.plot(y, u_anal_x, color='black', linestyle='-', linewidth=3, label='Аналитическое')
        plt.plot(y, u_adi_x, color=color, linestyle='--', linewidth=2, label='МПН')
        plt.plot(y, u_fs_x, color=color, linestyle=':', linewidth=2, label='Дробные шаги')
        
        plt.xlabel('y')
        plt.ylabel('u(x,y)')
        plt.title(f'{title}\nПри {label}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_errors_comparison(x, y, u_adi, u_fs, u_anal, title):
    plt.figure(figsize=(15, 10))
    
    y_indices = [len(y)//4, len(y)//2, 3*len(y)//4] 
    y_labels = [f'y = {y[i]:.3f}' for i in y_indices]
    colors_y = ['red', 'blue', 'green']
    
    for idx, (y_idx, color, label) in enumerate(zip(y_indices, colors_y, y_labels)):
        plt.subplot(2, 3, idx + 1)
        
        error_adi_y = [abs(u_adi[i][y_idx] - u_anal[i][y_idx]) for i in range(len(x))]
        error_fs_y = [abs(u_fs[i][y_idx] - u_anal[i][y_idx]) for i in range(len(x))]
        
        plt.plot(x, error_adi_y, color=color, linestyle='--', linewidth=2, label='МПН')
        plt.plot(x, error_fs_y, color=color, linestyle=':', linewidth=2, label='Дробные шаги')
        
        plt.xlabel('x')
        plt.ylabel('Погрешность')
        plt.title(f'Погрешности при {label}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
    
    x_indices = [len(x)//4, len(x)//2, 3*len(x)//4]
    x_labels = [f'x = {x[i]:.3f}' for i in x_indices]
    colors_x = ['orange', 'purple', 'brown']

    for idx, (x_idx, color, label) in enumerate(zip(x_indices, colors_x, x_labels)):
        plt.subplot(2, 3, idx + 4)
        
        error_adi_x = [abs(u_adi[x_idx][j] - u_anal[x_idx][j]) for j in range(len(y))]
        error_fs_x = [abs(u_fs[x_idx][j] - u_anal[x_idx][j]) for j in range(len(y))]
        
        plt.plot(y, error_adi_x, color=color, linestyle='--', linewidth=2, label='МПН')
        plt.plot(y, error_fs_x, color=color, linestyle=':', linewidth=2, label='Дробные шаги')
        
        plt.xlabel('y')
        plt.ylabel('Погрешность')
        plt.title(f'Погрешности при {label}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
    
    plt.tight_layout()
    plt.show()

def plot_3d_visualization(x, y, u_adi, u_fs, u_anal, title):
    from mpl_toolkits.mplot3d import Axes3D
    
    X, Y = np.meshgrid(x, y)
    U_anal_2d = np.array([[u_anal[i][j] for j in range(len(y))] for i in range(len(x))])
    U_adi_2d = np.array([[u_adi[i][j] for j in range(len(y))] for i in range(len(x))])
    U_fs_2d = np.array([[u_fs[i][j] for j in range(len(y))] for i in range(len(x))])
    
    error_adi_2d = np.abs(U_adi_2d - U_anal_2d)
    error_fs_2d = np.abs(U_fs_2d - U_anal_2d)
    
    fig = plt.figure(figsize=(18, 12))
    
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    surf1 = ax1.plot_surface(X, Y, U_anal_2d.T, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('u(x,y)')
    ax1.set_title(f'{title}\nАналитическое решение')
    fig.colorbar(surf1, ax=ax1, shrink=0.5)

    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    surf2 = ax2.plot_surface(X, Y, U_adi_2d.T, cmap='viridis', alpha=0.8)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('u(x,y)')
    ax2.set_title(f'{title}\nМПН')
    fig.colorbar(surf2, ax=ax2, shrink=0.5)

    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    surf3 = ax3.plot_surface(X, Y, U_fs_2d.T, cmap='viridis', alpha=0.8)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('u(x,y)')
    ax3.set_title(f'{title}\nДробные шаги')
    fig.colorbar(surf3, ax=ax3, shrink=0.5)

    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    surf4 = ax4.plot_surface(X, Y, error_adi_2d.T, cmap='hot', alpha=0.8)
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_zlabel('Погрешность')
    ax4.set_title(f'{title}\nПогрешность МПН')
    fig.colorbar(surf4, ax=ax4, shrink=0.5)

    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    surf5 = ax5.plot_surface(X, Y, error_fs_2d.T, cmap='hot', alpha=0.8)
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    ax5.set_zlabel('Погрешность')
    ax5.set_title(f'{title}\nПогрешность дробных шагов')
    fig.colorbar(surf5, ax=ax5, shrink=0.5)

    plt.tight_layout()
    plt.show()

def plot_error_evolution(errors_adi, errors_fs, title):
    times_adi = [err[0] for err in errors_adi]
    max_errors_adi = [err[1] for err in errors_adi]
    avg_errors_adi = [err[2] for err in errors_adi]
    
    times_fs = [err[0] for err in errors_fs]
    max_errors_fs = [err[1] for err in errors_fs]
    avg_errors_fs = [err[2] for err in errors_fs]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(times_adi, max_errors_adi, 'r-', linewidth=2, label='МПН')
    plt.plot(times_fs, max_errors_fs, 'b--', linewidth=2, label='Дробные шаги')
    plt.xlabel('Время')
    plt.ylabel('Максимальная погрешность')
    plt.title(f'{title}\nСравнение максимальных погрешностей')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.subplot(1, 2, 2)
    plt.plot(times_adi, avg_errors_adi, 'r-', linewidth=2, label='МПН')
    plt.plot(times_fs, avg_errors_fs, 'b--', linewidth=2, label='Дробные шаги')
    plt.xlabel('Время')
    plt.ylabel('Средняя погрешность')
    plt.title(f'{title}\nСравнение средних погрешностей')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.show()
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator

def study_grid_dependence(mu1=1, mu2=1, a=1, T=0.1):
    """
    Исследование зависимости погрешности от сеточных параметров
    """
    print("Исследование зависимости погрешности от сеточных параметров...")
    
    # Базовые параметры
    base_nx = 51
    base_ny = 51
    base_nt = 100
    
    Lx = (math.pi * mu1) / 2
    Ly = (math.pi * mu2) / 2
    
    # 1. Исследование зависимости от hx при фиксированном hy
    print("\n1. Зависимость от hx при фиксированном hy:")
    hy_fixed = Ly / (base_ny - 1)
    nx_values = [11, 21, 31, 41, 51, 61, 81, 101]
    hx_values = []
    errors_adi_hx = []
    errors_fs_hx = []
    
    for nx in nx_values:
        hx = Lx / (nx - 1)
        hx_values.append(hx)
        
        # Используем схему переменных направлений для оценки погрешности
        try:
            u_adi, errors_adi = scheme_variable_directions(mu1, mu2, a, nx, base_ny, base_nt, T)
            u_fs, errors_fs = scheme_fractional_steps(mu1, mu2, a, nx, base_ny, base_nt, T)
            
            x = [i * hx for i in range(nx)]
            y = [j * hy_fixed for j in range(base_ny)]
            u_anal = [[U(x[i], y[j], T, a, mu1, mu2) for j in range(base_ny)] for i in range(nx)]
            
            max_error_adi, avg_error_adi, _ = calc_error(u_adi, u_anal, nx, base_ny)
            max_error_fs, avg_error_fs, _ = calc_error(u_fs, u_anal, nx, base_ny)
            
            errors_adi_hx.append(avg_error_adi)
            errors_fs_hx.append(avg_error_fs)
            
            print(f"nx={nx:3d}, hx={hx:.3e}: ADI={avg_error_adi:.2e}, FS={avg_error_fs:.2e}")
            
        except Exception as e:
            print(f"Ошибка для nx={nx}: {e}")
            errors_adi_hx.append(np.nan)
            errors_fs_hx.append(np.nan)
    
    # 2. Исследование зависимости от hy при фиксированном hx
    print("\n2. Зависимость от hy при фиксированном hx:")
    hx_fixed = Lx / (base_nx - 1)
    ny_values = [11, 21, 31, 41, 51, 61, 81, 101]
    hy_values = []
    errors_adi_hy = []
    errors_fs_hy = []
    
    for ny in ny_values:
        hy = Ly / (ny - 1)
        hy_values.append(hy)
        
        try:
            u_adi, errors_adi = scheme_variable_directions(mu1, mu2, a, base_nx, ny, base_nt, T)
            u_fs, errors_fs = scheme_fractional_steps(mu1, mu2, a, base_nx, ny, base_nt, T)
            
            x = [i * hx_fixed for i in range(base_nx)]
            y = [j * hy for j in range(ny)]
            u_anal = [[U(x[i], y[j], T, a, mu1, mu2) for j in range(ny)] for i in range(base_nx)]
            
            max_error_adi, avg_error_adi, _ = calc_error(u_adi, u_anal, base_nx, ny)
            max_error_fs, avg_error_fs, _ = calc_error(u_fs, u_anal, base_nx, ny)
            
            errors_adi_hy.append(avg_error_adi)
            errors_fs_hy.append(avg_error_fs)
            
            print(f"ny={ny:3d}, hy={hy:.3e}: ADI={avg_error_adi:.2e}, FS={avg_error_fs:.2e}")
            
        except Exception as e:
            print(f"Ошибка для ny={ny}: {e}")
            errors_adi_hy.append(np.nan)
            errors_fs_hy.append(np.nan)
    
    # 3. Исследование зависимости при hx = hy
    print("\n3. Зависимость при hx = hy:")
    n_values = [11, 21, 31, 41, 51, 61, 81, 101]
    h_values = []
    errors_adi_h = []
    errors_fs_h = []
    
    for n in n_values:
        hx = Lx / (n - 1)
        hy = Ly / (n - 1)
        h_values.append(hx)  # используем hx, так как hx = hy
        
        try:
            u_adi, errors_adi = scheme_variable_directions(mu1, mu2, a, n, n, base_nt, T)
            u_fs, errors_fs = scheme_fractional_steps(mu1, mu2, a, n, n, base_nt, T)
            
            x = [i * hx for i in range(n)]
            y = [j * hy for j in range(n)]
            u_anal = [[U(x[i], y[j], T, a, mu1, mu2) for j in range(n)] for i in range(n)]
            
            max_error_adi, avg_error_adi, _ = calc_error(u_adi, u_anal, n, n)
            max_error_fs, avg_error_fs, _ = calc_error(u_fs, u_anal, n, n)
            
            errors_adi_h.append(avg_error_adi)
            errors_fs_h.append(avg_error_fs)
            
            print(f"n={n:3d}, h={hx:.3e}: ADI={avg_error_adi:.2e}, FS={avg_error_fs:.2e}")
            
        except Exception as e:
            print(f"Ошибка для n={n}: {e}")
            errors_adi_h.append(np.nan)
            errors_fs_h.append(np.nan)
    
    # 4. Исследование зависимости от tau
    print("\n4. Зависимость от tau:")
    base_n = 51
    nt_values = [10, 20, 50, 100, 200, 500, 1000]
    tau_values = []
    errors_adi_tau = []
    errors_fs_tau = []
    
    for nt in nt_values:
        tau = T / nt
        tau_values.append(tau)
        
        try:
            u_adi, errors_adi = scheme_variable_directions(mu1, mu2, a, base_n, base_n, nt, T)
            u_fs, errors_fs = scheme_fractional_steps(mu1, mu2, a, base_n, base_n, nt, T)
            
            hx = Lx / (base_n - 1)
            hy = Ly / (base_n - 1)
            x = [i * hx for i in range(base_n)]
            y = [j * hy for j in range(base_n)]
            u_anal = [[U(x[i], y[j], T, a, mu1, mu2) for j in range(base_n)] for i in range(base_n)]
            
            max_error_adi, avg_error_adi, _ = calc_error(u_adi, u_anal, base_n, base_n)
            max_error_fs, avg_error_fs, _ = calc_error(u_fs, u_anal, base_n, base_n)
            
            errors_adi_tau.append(avg_error_adi)
            errors_fs_tau.append(avg_error_fs)
            
            print(f"nt={nt:4d}, tau={tau:.3e}: ADI={avg_error_adi:.2e}, FS={avg_error_fs:.2e}")
            
        except Exception as e:
            print(f"Ошибка для nt={nt}: {e}")
            errors_adi_tau.append(np.nan)
            errors_fs_tau.append(np.nan)
    
    # Построение графиков
    plot_grid_dependence_study(
        hx_values, errors_adi_hx, errors_fs_hx,
        hy_values, errors_adi_hy, errors_fs_hy, 
        h_values, errors_adi_h, errors_fs_h,
        tau_values, errors_adi_tau, errors_fs_tau,
        mu1, mu2
    )
    
    return {
        'hx': (hx_values, errors_adi_hx, errors_fs_hx),
        'hy': (hy_values, errors_adi_hy, errors_fs_hy),
        'h': (h_values, errors_adi_h, errors_fs_h),
        'tau': (tau_values, errors_adi_tau, errors_fs_tau)
    }

def plot_grid_dependence_study(hx_values, errors_adi_hx, errors_fs_hx,
                              hy_values, errors_adi_hy, errors_fs_hy,
                              h_values, errors_adi_h, errors_fs_h,
                              tau_values, errors_adi_tau, errors_fs_tau,
                              mu1, mu2):

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Исследование зависимости погрешности от сеточных параметров (μ₁={mu1}, μ₂={mu2})', 
                 fontsize=14, fontweight='bold')
    
    # 1. Зависимость от hx
    ax1 = axes[0, 0]
    valid_hx = [(h, e_adi, e_fs) for h, e_adi, e_fs in zip(hx_values, errors_adi_hx, errors_fs_hx) 
                if not (np.isnan(e_adi) or np.isnan(e_fs))]
    if valid_hx:
        h_vals, e_adi_vals, e_fs_vals = zip(*valid_hx)
        ax1.loglog(h_vals, e_adi_vals, 'ro-', linewidth=2, markersize=6, label='МПН')
        ax1.loglog(h_vals, e_fs_vals, 'bs-', linewidth=2, markersize=6, label='Дробные шаги')
        
        if len(h_vals) > 1:
            # Линия O(h²)
            ref_h2 = [e_adi_vals[0] * (h/h_vals[0])**2 for h in h_vals]
            ax1.loglog(h_vals, ref_h2, 'k--', linewidth=1, alpha=0.7, label='O(h²)')
    
    ax1.set_xlabel('Шаг по x (hx)')
    ax1.set_ylabel('Средняя погрешность')
    ax1.set_title('Зависимость от hx (hy фиксирован)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Зависимость от hy
    ax2 = axes[0, 1]
    valid_hy = [(h, e_adi, e_fs) for h, e_adi, e_fs in zip(hy_values, errors_adi_hy, errors_fs_hy) 
                if not (np.isnan(e_adi) or np.isnan(e_fs))]
    if valid_hy:
        h_vals, e_adi_vals, e_fs_vals = zip(*valid_hy)
        ax2.loglog(h_vals, e_adi_vals, 'ro-', linewidth=2, markersize=6, label='МПН')
        ax2.loglog(h_vals, e_fs_vals, 'bs-', linewidth=2, markersize=6, label='Дробные шаги')
        
        # Линия O(h²)
        if len(h_vals) > 1:
            ref_h2 = [e_adi_vals[0] * (h/h_vals[0])**2 for h in h_vals]
            ax2.loglog(h_vals, ref_h2, 'k--', linewidth=1, alpha=0.7, label='O(h²)')
    
    ax2.set_xlabel('Шаг по y (hy)')
    ax2.set_ylabel('Средняя погрешность')
    ax2.set_title('Зависимость от hy (hx фиксирован)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Зависимость при hx = hy
    ax3 = axes[1, 0]
    valid_h = [(h, e_adi, e_fs) for h, e_adi, e_fs in zip(h_values, errors_adi_h, errors_fs_h) 
               if not (np.isnan(e_adi) or np.isnan(e_fs))]
    if valid_h:
        h_vals, e_adi_vals, e_fs_vals = zip(*valid_h)
        ax3.loglog(h_vals, e_adi_vals, 'ro-', linewidth=2, markersize=6, label='МПН')
        ax3.loglog(h_vals, e_fs_vals, 'bs-', linewidth=2, markersize=6, label='Дробные шаги')
        
        # Линия O(h²)
        if len(h_vals) > 1:
            ref_h2 = [e_adi_vals[0] * (h/h_vals[0])**2 for h in h_vals]
            ax3.loglog(h_vals, ref_h2, 'k--', linewidth=1, alpha=0.7, label='O(h²)')
    
    ax3.set_xlabel('Шаг сетки (h = hx = hy)')
    ax3.set_ylabel('Средняя погрешность')
    ax3.set_title('Зависимость при hx = hy')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Зависимость от tau
    ax4 = axes[1, 1]
    valid_tau = [(tau, e_adi, e_fs) for tau, e_adi, e_fs in zip(tau_values, errors_adi_tau, errors_fs_tau) 
                 if not (np.isnan(e_adi) or np.isnan(e_fs))]
    if valid_tau:
        tau_vals, e_adi_vals, e_fs_vals = zip(*valid_tau)
        ax4.loglog(tau_vals, e_adi_vals, 'ro-', linewidth=2, markersize=6, label='МПН')
        ax4.loglog(tau_vals, e_fs_vals, 'bs-', linewidth=2, markersize=6, label='Дробные шаги')
        
        # Линия O(τ)
        if len(tau_vals) > 1:
            ref_tau1 = [e_adi_vals[0] * (tau/tau_vals[0]) for tau in tau_vals]
            ax4.loglog(tau_vals, ref_tau1, 'k--', linewidth=1, alpha=0.7, label='O(τ)')
    
    ax4.set_xlabel('Временной шаг (τ)')
    ax4.set_ylabel('Средняя погрешность')
    ax4.set_title('Зависимость от временного шага')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    test_cases = [
        (1, 1, "μ₁=1, μ₂=1"),
        (2, 1, "μ₁=2, μ₂=1"), 
        (1, 2, "μ₁=1, μ₂=2")
    ]
    
    a = 1
    nx = 51
    ny = 51
    nt = 50
    T = 0.1

    for mu1, mu2, case_name in test_cases:
        print(f"\n{'='*60}")
        print(f"Тестирование случая: {case_name}")
        print(f"{'='*60}")
        
        Lx = (math.pi * mu1) / 2
        Ly = (math.pi * mu2) / 2
        hx = Lx / (nx - 1)
        hy = Ly / (ny - 1)
        tau = T / (nt - 1)

        x = [i * hx for i in range(nx)]
        y = [j * hy for j in range(ny)]
        
        print(f"Параметры: μ1 = {mu1}, μ2 = {mu2}, a = {a}")
        print(f"Область: Lx = {Lx:.3f}, Ly = {Ly:.3f}")
        print(f"Сетка: {nx} × {ny}, шаги: hx = {hx:.3e}, hy = {hy:.3e}")
        print(f"Временные параметры: T = {T}, τ = {tau:.3e}, шагов: {nt}")

        try:
            u_adi, errors_adi = scheme_variable_directions(mu1, mu2, a, nx, ny, nt, T)
            u_fs, errors_fs = scheme_fractional_steps(mu1, mu2, a, nx, ny, nt, T)
          
            u_anal = [[U(x[i], y[j], T, a, mu1, mu2) for j in range(ny)] for i in range(nx)]

            max_error_adi, avg_error_adi, _ = calc_error(u_adi, u_anal, nx, ny)
            max_error_fs, avg_error_fs, _ = calc_error(u_fs, u_anal, nx, ny)
            
            print(f"\nИтоговые погрешности в момент T = {T}:")
            print("Метод переменных направлений:")
            print(f"  Максимальная погрешность: {max_error_adi:.2e}")
            print(f"  Средняя погрешность: {avg_error_adi:.2e}")
            print("Метод дробных шагов:")
            print(f"  Максимальная погрешность: {max_error_fs:.2e}")
            print(f"  Средняя погрешность: {avg_error_fs:.2e}")

            plot_solutions_comparison(x, y, u_adi, u_fs, u_anal, f"Сравнение методов ({case_name})")
            plot_errors_comparison(x, y, u_adi, u_fs, u_anal, f"Сравнение методов ({case_name})")
            plot_3d_visualization(x, y, u_adi, u_fs, u_anal, f"Сравнение методов ({case_name})")
            plot_error_evolution(errors_adi, errors_fs, f"Сравнение методов ({case_name})")
            
        except Exception as e:
            print(f"Ошибка при решении для случая {case_name}: {e}")
            continue

    print("\n" + "="*80)
    print("ИССЛЕДОВАНИЕ ЗАВИСИМОСТИ ОТ СЕТОЧНЫХ ПАРАМЕТРОВ")
    print("="*80)
    
    study_grid_dependence(mu1=1, mu2=1, a=1, T=0.1)

if __name__ == "__main__":
    main()