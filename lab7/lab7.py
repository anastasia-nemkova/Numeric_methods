import math
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def U(x, y):
    return x * x - y * y

def init_grid(nx, ny, lx, ly):
    u = [[0.0 for _ in range(ny)] for _ in range(nx)]

    hx = lx / (nx - 1)
    x = [i * hx for i in range(nx)]

    hy = ly / (ny - 1)
    y = [j * hy for j in range(ny)]

    return u, x, y, hx, hy

def bound_cond_first_order(u, x, y, nx, ny):
    for j in range(ny):
        u[nx - 1][j] = 1 - y[j] * y[j]

    for i in range(nx):
        u[i][ny - 1] = x[i] * x[i] - 1

    for j in range(ny):
        u[0][j] = u[1][j]

    for i in range(nx):
        u[i][0] = u[i][1] 

def bound_cond_second_order(u, x, y, nx, ny, hx, hy):
    for j in range(ny):
        u[nx - 1][j] = 1 - y[j] * y[j]

    for i in range(nx):
        u[i][ny - 1] = x[i] * x[i] - 1

    for j in range(ny):
        u[0][j] = (4 * u[1][j] - u[2][j]) / 3

    for i in range(nx):
        u[i][0] = (4 * u[i][1] - u[i][2]) / 3

def update_boundary_conditions_first_order(u, nx, ny):
    for j in range(ny):
        u[0][j] = u[1][j]
    for i in range(nx):
        u[i][0] = u[i][1]

def update_boundary_conditions_second_order(u, nx, ny):
    for j in range(ny):
        u[0][j] = (4 * u[1][j] - u[2][j]) / 3
    for i in range(nx):
        u[i][0] = (4 * u[i][1] - u[i][2]) / 3

def calc_norm(u_new, u, nx, ny):
    norma = 0.0
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            diff = abs(u_new[i][j] - u[i][j])
            if diff > norma:
                norma = diff
    return norma

def method_Libman(u, hx, hy, nx, ny, boundary_type='first_order', max_iter=10000, eps=1e-6):
    u_new = [[u[i][j] for j in range(ny)] for i in range(nx)]
    iter = 0

    for _ in range(max_iter):
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                u_new[i][j] = 0.5 * (hx * hx * hy * hy) / (hx * hx + hy * hy) * ((u[i + 1][j] + u[i - 1][j])/(hx * hx) + (u[i][j + 1] + u[i][j - 1])/(hy * hy))

        norma = calc_norm(u_new, u, nx, ny)

        for i in range(1, nx):
            for j in range(1, ny):
                u[i][j] = u_new[i][j]

        if boundary_type == 'first_order':
            update_boundary_conditions_first_order(u, nx, ny)
        else:
            update_boundary_conditions_second_order(u, nx, ny)

        iter += 1
        if norma < eps:
            break
    return u, iter

def method_Seidel(u, hx, hy, nx, ny, boundary_type='first_order', max_iter=10000, eps=1e-6):
    u_old = [[u[i][j] for j in range(ny)] for i in range(nx)]
    iter = 0

    for _ in range(max_iter):
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                u_old[i][j] = u[i][j]

        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                u[i][j] =  0.5 * (hx * hx * hy * hy) / (hx * hx + hy * hy) * ((u[i + 1][j] + u[i - 1][j])/(hx * hx) + (u[i][j + 1] + u[i][j - 1])/(hy * hy))

        if boundary_type == 'first_order':
            update_boundary_conditions_first_order(u, nx, ny)
        else:
            update_boundary_conditions_second_order(u, nx, ny)

        norm = calc_norm(u, u_old, nx, ny)

        iter += 1
        if norm < eps:
            break
    return u, iter

def method_SOR(u, hx, hy, nx, ny, boundary_type='first_order', max_iter=10000, eps=1e-6, omega=1.5):
    u_old = [[u[i][j] for j in range(ny)] for i in range(nx)]
    iter = 0

    for _ in range(max_iter):
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                u_old[i][j] = u[i][j]

        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                u_sor =  0.5 * (hx * hx * hy * hy) / (hx * hx + hy * hy) * ((u[i + 1][j] + u[i - 1][j])/(hx * hx) + (u[i][j + 1] + u[i][j - 1])/(hy * hy))
                u[i][j] = u_old[i][j] + omega * (u_sor - u_old[i][j])

        if boundary_type == 'first_order':
            update_boundary_conditions_first_order(u, nx, ny)
        else:
            update_boundary_conditions_second_order(u, nx, ny)

        norm = calc_norm(u, u_old, nx, ny)

        iter += 1
        if norm < eps:
            break
    return u, iter

def calc_error(u, x, y, nx, ny):
    max_error = 0.0
    rmse_error = 0.0
    count = 0
    
    for i in range(nx):
        for j in range(ny):
            exact = U(x[i], y[j])
            error = abs(u[i][j] - exact)
            if error > max_error:
                max_error = error
            rmse_error += error * error
            count += 1
    
    rmse_error = math.sqrt(rmse_error / count) if count > 0 else 0.0
    return max_error, rmse_error

def plot_3d_solution(u, x, y, nx, ny, title):
    X, Y = np.meshgrid(x, y, indexing='ij')
    Z = np.array(u)
    
    fig = plt.figure(figsize=(12, 5))
    
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('u(x,y)')
    ax1.set_title(f'{title} - 3D поверхность')
    
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(X, Y, Z, levels=20, cmap='viridis')
    plt.colorbar(contour, ax=ax2)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title(f'{title} - Контурный график')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_3d_error(u_numeric, u_exact, x, y, nx, ny, title):
    X, Y = np.meshgrid(x, y, indexing='ij')
    error = np.abs(np.array(u_numeric) - np.array(u_exact))
    
    fig = plt.figure(figsize=(12, 5))
    
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(X, Y, error, cmap='hot', alpha=0.8)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('Погрешность')
    ax1.set_title(f'{title} - 3D погрешность')
    
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(X, Y, error, levels=20, cmap='hot')
    plt.colorbar(contour, ax=ax2)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title(f'{title} - Контур погрешности')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_solution(u_lib, u_seid, u_sor, u_analytical, x, y, nx, ny):
    y_indices = [
        int(0.25 * (ny - 1)),
        int(0.5 * (ny - 1)),
        int(0.75 * (ny - 1))
    ]
    
    y_values = [y[idx] for idx in y_indices]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, (ax, y_idx, y_val) in enumerate(zip(axes, y_indices, y_values)):
        u_analytical_y = [u_analytical[i][y_idx] for i in range(nx)]
        u_lib_y = [u_lib[i][y_idx] for i in range(nx)]
        u_seid_y = [u_seid[i][y_idx] for i in range(nx)]
        u_sor_y = [u_sor[i][y_idx] for i in range(nx)]
        
        ax.plot(x, u_analytical_y, 'k-', linewidth=3, label='Аналитическое', alpha=0.8)
        ax.plot(x, u_lib_y, 'r--', linewidth=2, label='Либман')
        ax.plot(x, u_seid_y, 'g-.', linewidth=2, label='Зейдель')
        ax.plot(x, u_sor_y, 'b:', linewidth=2, label='Верхняя релаксация')
        
        ax.set_xlabel('x')
        ax.set_ylabel('u(x, y)')
        ax.set_title(f'y = {y_val:.2f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Сравнение методов с аналитическим решением', fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_errors(u_lib, u_seid, u_sor, u_analytical, x, y, nx, ny):
    y_indices = [
        int(0.25 * (ny - 1)),
        int(0.5 * (ny - 1)),
        int(0.75 * (ny - 1))
    ]
    
    y_values = [y[idx] for idx in y_indices]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, (ax, y_idx, y_val) in enumerate(zip(axes, y_indices, y_values)):
        errors_lib = [abs(u_lib[i][y_idx] - u_analytical[i][y_idx]) for i in range(nx)]
        errors_seid = [abs(u_seid[i][y_idx] - u_analytical[i][y_idx]) for i in range(nx)]
        errors_sor = [abs(u_sor[i][y_idx] - u_analytical[i][y_idx]) for i in range(nx)]
        
        ax.plot(x, errors_lib, 'r--', linewidth=2, label='Либман')
        ax.plot(x, errors_seid, 'g-.', linewidth=2, label='Зейдель')
        ax.plot(x, errors_sor, 'b:', linewidth=2, label='Верхняя релаксация')
        
        ax.set_xlabel('x')
        ax.set_ylabel('Погрешность')
        ax.set_title(f'Погрешности методов при y = {y_val:.2f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Погрешности численных методов', fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_comparison_boundary_conditions():
    lx, ly = 1.0, 1.0
    nx, ny = 51, 51

    u, x, y, hx, hy = init_grid(nx, ny, lx, ly)
    u_exact = [[U(x[i], y[j]) for j in range(ny)] for i in range(nx)]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    boundary_types = ['first_order', 'second_order']
    boundary_names = ['Первый порядок', 'Второй порядок']
    
    for row, (boundary_type, boundary_name) in enumerate(zip(boundary_types, boundary_names)):
        u_init = [[u[i][j] for j in range(ny)] for i in range(nx)]
        if boundary_type == 'first_order':
            bound_cond_first_order(u_init, x, y, nx, ny)
        else:
            bound_cond_second_order(u_init, x, y, nx, ny, hx, hy)

        u_lib = [[u_init[i][j] for j in range(ny)] for i in range(nx)]
        u_lib, lib_iter = method_Libman(u_lib, hx, hy, nx, ny, boundary_type)
        max_err_lib, rmse_lib = calc_error(u_lib, x, y, nx, ny)
 
        u_seid = [[u_init[i][j] for j in range(ny)] for i in range(nx)]
        u_seid, seid_iter = method_Seidel(u_seid, hx, hy, nx, ny, boundary_type)
        max_err_seid, rmse_seid = calc_error(u_seid, x, y, nx, ny)
 
        u_sor = [[u_init[i][j] for j in range(ny)] for i in range(nx)]
        u_sor, sor_iter = method_SOR(u_sor, hx, hy, nx, ny, boundary_type)
        max_err_sor, rmse_sor = calc_error(u_sor, x, y, nx, ny)
        
        print(f"\n--- {boundary_name} ---")
        print(f"Либман: {max_err_lib:.2e} (итераций: {lib_iter})")
        print(f"Зейдель: {max_err_seid:.2e} (итераций: {seid_iter})")
        print(f"SOR: {max_err_sor:.2e} (итераций: {sor_iter})")

        y_indices = [
            int(0.25 * (ny - 1)),
            int(0.5 * (ny - 1)),
            int(0.75 * (ny - 1))
        ]
        
        y_values = [y[idx] for idx in y_indices]
        
        for col, (y_idx, y_val) in enumerate(zip(y_indices, y_values)):
            ax = axes[row, col]

            u_analytical_y = [u_exact[i][y_idx] for i in range(nx)]
            u_lib_y = [u_lib[i][y_idx] for i in range(nx)]
            u_seid_y = [u_seid[i][y_idx] for i in range(nx)]
            u_sor_y = [u_sor[i][y_idx] for i in range(nx)]
            
            ax.plot(x, u_analytical_y, 'k-', linewidth=3, label='Аналитическое', alpha=0.8)
            ax.plot(x, u_lib_y, 'r--', linewidth=2, label='Либман')
            ax.plot(x, u_seid_y, 'g-.', linewidth=2, label='Зейдель')
            ax.plot(x, u_sor_y, 'b:', linewidth=2, label='Верхняя релаксация')
            
            ax.set_xlabel('x')
            ax.set_ylabel('u(x, y)')
            ax.set_title(f'{boundary_name}, y = {y_val:.2f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.suptitle('Сравнение методов при разных аппроксимациях граничных условий', fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_grid_convergence():
    lx, ly = 1.0, 1.0
    grid_sizes = [11, 21, 31, 41, 51, 61, 71]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    fixed_ny = 51
    
    libman_errors_hx = []
    seidel_errors_hx = []
    sor_errors_hx = []
    hx_values = []
    
    for nx in grid_sizes:
        ny = fixed_ny
        u, x, y, hx, hy = init_grid(nx, ny, lx, ly)
        bound_cond_first_order(u, x, y, nx, ny)
        
        hx_values.append(hx)

        u_lib = [[u[i][j] for j in range(ny)] for i in range(nx)]
        u_lib, _ = method_Libman(u_lib, hx, hy, nx, ny)
        max_err_lib, _ = calc_error(u_lib, x, y, nx, ny)
        libman_errors_hx.append(max_err_lib)

        u_seid = [[u[i][j] for j in range(ny)] for i in range(nx)]
        u_seid, _ = method_Seidel(u_seid, hx, hy, nx, ny)
        max_err_seid, _ = calc_error(u_seid, x, y, nx, ny)
        seidel_errors_hx.append(max_err_seid)

        u_sor = [[u[i][j] for j in range(ny)] for i in range(nx)]
        u_sor, _ = method_SOR(u_sor, hx, hy, nx, ny)
        max_err_sor, _ = calc_error(u_sor, x, y, nx, ny)
        sor_errors_hx.append(max_err_sor)

    fixed_nx = 51
    libman_errors_hy = []
    seidel_errors_hy = []
    sor_errors_hy = []
    hy_values = []
    
    for ny in grid_sizes:
        nx = fixed_nx
        u, x, y, hx, hy = init_grid(nx, ny, lx, ly)
        bound_cond_first_order(u, x, y, nx, ny)
        
        hy_values.append(hy)

        u_lib = [[u[i][j] for j in range(ny)] for i in range(nx)]
        u_lib, _ = method_Libman(u_lib, hx, hy, nx, ny)
        max_err_lib, _ = calc_error(u_lib, x, y, nx, ny)
        libman_errors_hy.append(max_err_lib)

        u_seid = [[u[i][j] for j in range(ny)] for i in range(nx)]
        u_seid, _ = method_Seidel(u_seid, hx, hy, nx, ny)
        max_err_seid, _ = calc_error(u_seid, x, y, nx, ny)
        seidel_errors_hy.append(max_err_seid)

        u_sor = [[u[i][j] for j in range(ny)] for i in range(nx)]
        u_sor, _ = method_SOR(u_sor, hx, hy, nx, ny)
        max_err_sor, _ = calc_error(u_sor, x, y, nx, ny)
        sor_errors_hy.append(max_err_sor)
    
    libman_errors_square = []
    seidel_errors_square = []
    sor_errors_square = []
    h_values = []
    
    for n in grid_sizes:
        nx = ny = n
        u, x, y, hx, hy = init_grid(nx, ny, lx, ly)
        bound_cond_first_order(u, x, y, nx, ny)
        
        h_values.append(hx)

        u_lib = [[u[i][j] for j in range(ny)] for i in range(nx)]
        u_lib, _ = method_Libman(u_lib, hx, hy, nx, ny)
        max_err_lib, _ = calc_error(u_lib, x, y, nx, ny)
        libman_errors_square.append(max_err_lib)

        u_seid = [[u[i][j] for j in range(ny)] for i in range(nx)]
        u_seid, _ = method_Seidel(u_seid, hx, hy, nx, ny)
        max_err_seid, _ = calc_error(u_seid, x, y, nx, ny)
        seidel_errors_square.append(max_err_seid)

        u_sor = [[u[i][j] for j in range(ny)] for i in range(nx)]
        u_sor, _ = method_SOR(u_sor, hx, hy, nx, ny)
        max_err_sor, _ = calc_error(u_sor, x, y, nx, ny)
        sor_errors_square.append(max_err_sor)

    ax1.plot(hx_values, libman_errors_hx, 'ro-', linewidth=2, markersize=6, label='Метод Либмана')
    ax1.plot(hx_values, seidel_errors_hx, 'gs-', linewidth=2, markersize=6, label='Метод Зейделя')
    ax1.plot(hx_values, sor_errors_hx, 'b^-', linewidth=2, markersize=6, label='Метод верхней релаксации')
    ax1.set_xlabel('Шаг сетки по x (hx)')
    ax1.set_ylabel('Максимальная погрешность')
    ax1.set_title('Фиксированное hy = 0.02')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('linear')

    ax2.plot(hy_values, libman_errors_hy, 'ro-', linewidth=2, markersize=6, label='Метод Либмана')
    ax2.plot(hy_values, seidel_errors_hy, 'gs-', linewidth=2, markersize=6, label='Метод Зейделя')
    ax2.plot(hy_values, sor_errors_hy, 'b^-', linewidth=2, markersize=6, label='Метод верхней релаксации')
    ax2.set_xlabel('Шаг сетки по y (hy)')
    ax2.set_ylabel('Максимальная погрешность')
    ax2.set_title('Фиксированное hx = 0.02')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('linear')

    ax3.plot(h_values, libman_errors_square, 'ro-', linewidth=2, markersize=6, label='Метод Либмана')
    ax3.plot(h_values, seidel_errors_square, 'gs-', linewidth=2, markersize=6, label='Метод Зейделя')
    ax3.plot(h_values, sor_errors_square, 'b^-', linewidth=2, markersize=6, label='Метод верхней релаксации')
    ax3.set_xlabel('Шаг сетки hx = hy')
    ax3.set_ylabel('Максимальная погрешность')
    ax3.set_title('Квадратная сетка (hx = hy)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('linear')
    
    plt.suptitle('Зависимость погрешности от сеточных параметров', fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_iterations_convergence():
    lx, ly = 1.0, 1.0
    grid_sizes = [11, 21, 31, 41, 51]
    
    libman_iters = []
    seidel_iters = []
    sor_iters = []
    
    for n in grid_sizes:
        nx = ny = n
        u, x, y, hx, hy = init_grid(nx, ny, lx, ly)
        bound_cond_first_order(u, x, y, nx, ny)

        u_lib = [[u[i][j] for j in range(ny)] for i in range(nx)]
        _, iter_lib = method_Libman(u_lib, hx, hy, nx, ny)
        libman_iters.append(iter_lib)

        u_seid = [[u[i][j] for j in range(ny)] for i in range(nx)]
        _, iter_seid = method_Seidel(u_seid, hx, hy, nx, ny)
        seidel_iters.append(iter_seid)
 
        u_sor = [[u[i][j] for j in range(ny)] for i in range(nx)]
        _, iter_sor = method_SOR(u_sor, hx, hy, nx, ny)
        sor_iters.append(iter_sor)

    plt.figure(figsize=(10, 6))
    plt.plot(grid_sizes, libman_iters, 'ro-', linewidth=2, markersize=6, label='Метод Либмана')
    plt.plot(grid_sizes, seidel_iters, 'gs-', linewidth=2, markersize=6, label='Метод Зейделя')
    plt.plot(grid_sizes, sor_iters, 'b^-', linewidth=2, markersize=6, label='Метод верхней релаксации')
    
    plt.xlabel('Количество узлов сетки N')
    plt.ylabel('Число итераций')
    plt.title('Зависимость числа итераций от размера сетки')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('linear')
    plt.show()

def print_results(method_name, iterations, max_error, rms_error):
    print(f"\n{method_name}:")
    print(f"  Итераций: {iterations}")
    print(f"  Максимальная погрешность: {max_error:.6e}")
    print(f"  RMSE: {rms_error:.6e}")

def main():
    lx = 1.0
    nx = 51

    ly = 1.0
    ny = 51

    u, x, y, hx, hy = init_grid(nx, ny, lx, ly)
    bound_cond_first_order(u, x, y, nx, ny)

    u_lib = [[u[i][j] for j in range(ny)] for i in range(nx)]
    u_lib, lib_iter = method_Libman(u_lib, hx, hy, nx, ny)
    max_err_lib, rmse_lib = calc_error(u_lib, x, y, nx, ny)
    print_results("Метод Либмана", lib_iter, max_err_lib, rmse_lib)

    u_seid = [[u[i][j] for j in range(ny)] for i in range(nx)]
    u_seid, seid_iter = method_Seidel(u_seid, hx, hy, nx, ny)
    max_err_seid, rmse_seid = calc_error(u_seid, x, y, nx, ny)
    print_results("Метод Зейделя", seid_iter, max_err_seid, rmse_seid)

    u_sor = [[u[i][j] for j in range(ny)] for i in range(nx)]
    u_sor, sor_iter = method_SOR(u_sor, hx, hy, nx, ny)
    max_err_sor, rmse_sor = calc_error(u_sor, x, y, nx, ny)
    print_results("Метод верхней релаксации", sor_iter, max_err_sor, rmse_sor)

    u_exact = [[U(x[i], y[j]) for j in range(ny)] for i in range(nx)]
    
    plot_solution(u_lib, u_seid, u_sor, u_exact, x, y, nx, ny)
    plot_errors(u_lib, u_seid, u_sor, u_exact, x, y, nx, ny)
    
    plot_3d_solution(u_exact, x, y, nx, ny, "Аналитическое решение")
    plot_3d_solution(u_lib, x, y, nx, ny, "Метод Либмана")
    plot_3d_solution(u_seid, x, y, nx, ny, "Метод Зейделя")
    plot_3d_solution(u_sor, x, y, nx, ny, "Метод верхней релаксации")
    
    plot_3d_error(u_lib, u_exact, x, y, nx, ny, "Погрешность метода Либмана")
    plot_3d_error(u_seid, u_exact, x, y, nx, ny, "Погрешность метода Зейделя")
    plot_3d_error(u_sor, u_exact, x, y, nx, ny, "Погрешность метода верхней релаксации")

    plot_grid_convergence()
    plot_iterations_convergence()

if __name__ == "__main__":
    main()