import math
import matplotlib.pyplot as plt
import numpy as np

def U(x, t, a):
    return np.sin(x - a * t) + np.cos(x + a * t)

def init_cond(x):
    return math.sin(x) + math.cos(x)

def init_cond2(x, a):
    return -a * (math.sin(x) + math.cos(x))

def init_cond_xx(x):
    return -math.sin(x) - math.cos(x)

def calculate_error(u_numerical, x_points, t, a, N):
    error = 0.0
    for i in range(N + 1):
        exact = U(x_points[i], t, a)
        error += (u_numerical[i] - exact) ** 2
    return math.sqrt(error / (N + 1))

def explicit_scheme(L, N, T, K, a, bc_type, init_approx):
    h = L / N
    tao = T / K
    sigma = (a * tao / h) ** 2

    if sigma > 1:
        print("Схема может быть неустойчивой")

    u_prev = [0.0] * (N + 1)
    u_curr = [0.0] * (N + 1)
    u_next = [0.0] * (N + 1)

    x_points = [i * h for i in range(N + 1)]

    for i in range(N + 1):
        u_curr[i] = init_cond(x_points[i])

    if init_approx == 'first_order':
        for i in range(N + 1):
            u_prev[i] = u_curr[i] - tao * init_cond2(x_points[i], a)
    elif init_approx == 'second_order':
        for i in range(N + 1):
            u_prev[i] = u_curr[i] - tao * init_cond2(x_points[i], a) + 0.5 * (tao * a) ** 2 * init_cond_xx(x_points[i])

    errors = []
    solutions = []
    
    for n in range(1, K + 1):
        t = n * tao
        
        for i in range(1, N):
            u_next[i] = 2 * u_curr[i] - u_prev[i] + sigma * (u_curr[i + 1] - 2 * u_curr[i] + u_curr[i - 1])
        
        if bc_type == 'two_point_first':
            u_next[0] = u_next[1] / (1 + h)
        elif bc_type == 'three_point_second':
            u_next[0] = (4 * u_next[1] - u_next[2]) / (3 + 2 * h)
        elif bc_type == 'two_point_second':
            u_next[0] = (2 * (1 - sigma) * u_curr[0] + 2 * sigma * u_curr[1] - u_prev[0]) / (1 + 2 * sigma * h)

        # Граничные условия ux(pi,t) - u(pi,t) = 0
        if bc_type == 'two_point_first':
            u_next[N] = u_next[N - 1] / (1 - h)
        elif bc_type == 'three_point_second':
            u_next[N] = (4 * u_next[N - 1] - u_next[N - 2]) / (3 - 2 * h)
        elif bc_type == 'two_point_second':
            u_next[N] = (2 * (1 - sigma) * u_curr[N] + 2 * sigma * u_curr[N - 1] - u_prev[N]) / (1 - 2 * sigma * h)

        error = calculate_error(u_next, x_points, t, a, N)
        errors.append((t, error))
        solutions.append((t, u_next.copy()))
        
        u_prev, u_curr, u_next = u_curr, u_next, u_prev
    
    return errors, solutions

def tridiagonal_matrix_algorithm(a, b, c, d):
    size = len(a)
    p = [0.0] * size
    q = [0.0] * size
    
    p[0] = -c[0] / b[0]
    q[0] = d[0] / b[0]
    
    for i in range(1, size):
        denominator = b[i] + a[i] * p[i - 1]
        p[i] = -c[i] / denominator
        q[i] = (d[i] - a[i] * q[i - 1]) / denominator
    
    x = [0.0] * size
    x[size - 1] = q[size - 1]
    
    for i in range(size - 2, -1, -1):
        x[i] = p[i] * x[i + 1] + q[i]
    
    return x

def implicit_scheme(L, N, T, K, a, bc_type, init_approx):
    h = L / N
    tau = T / K
    sigma = (a * tau / h) ** 2


    u_prev = [0.0] * (N + 1)
    u_curr = [0.0] * (N + 1)
    u_next = [0.0] * (N + 1)

    x_points = [i * h for i in range(N + 1)]

    for i in range(N + 1):
        u_curr[i] = init_cond(x_points[i])

    if init_approx == 'first_order':
        for i in range(N + 1):
            u_prev[i] = u_curr[i] - tau * init_cond2(x_points[i], a)
    elif init_approx == 'second_order':
        for i in range(N + 1):
            u_prev[i] = u_curr[i] - tau * init_cond2(x_points[i], a) + 0.5 * (tau * a) ** 2 * init_cond_xx(x_points[i])

    errors = []
    solutions = []
        
    for k in range(1, K + 1):
        t = k * tau
        
        a_coeffs = [0.0] * (N + 1)
        b_coeffs = [0.0] * (N + 1) 
        c_coeffs = [0.0] * (N + 1)
        d_coeffs = [0.0] * (N + 1)

        for i in range(1, N):
            a_coeffs[i] = -sigma
            b_coeffs[i] = 1 + 2 * sigma
            c_coeffs[i] = -sigma
            d_coeffs[i] = 2 * u_curr[i] - u_prev[i]

        if bc_type == 'two_point_first':
            b_coeffs[0] = 1 + 1 / h
            c_coeffs[0] = -1 / h
            d_coeffs[0] = 0

            a_coeffs[N] = -1 / h
            b_coeffs[N] = -1 + 1 / h
            d_coeffs[N] = 0
            
        elif bc_type == 'three_point_second':
            b_coeffs[0] = (a / h + h / (tau * 2)) + 1
            c_coeffs[0] =  -a / h
            d_coeffs[0] = u_prev[0] * h / (tau * 2)
 
            a_coeffs[N] =  -a / h
            b_coeffs[N] = (a / h + h / (tau * 2)) - 1
            d_coeffs[N] = u_prev[N] * h / (tau * 2)
            
        elif bc_type == 'two_point_second':
            c_coeffs[0] = -2 / h - b_coeffs[1] / (2 * h * c_coeffs[1])
            b_coeffs[0] = (3 / (2 * h) + 1) - a_coeffs[1] / (2 * h * c_coeffs[1])
            d_coeffs[0] = -d_coeffs[1] * (1 / (2 * h * c_coeffs[1]))

            a_coeffs[N] = (2 / h) + b_coeffs[N - 1] / (h * 2 * a_coeffs[N - 1])
            b_coeffs[N] = (-3 / (2 * h) + 1) + c_coeffs[N - 1] / (h * 2 * a_coeffs[N - 1])
            d_coeffs[N] = d_coeffs[N - 1] / (h * 2 * a_coeffs[N - 1])

        u_next = tridiagonal_matrix_algorithm(a_coeffs, b_coeffs, c_coeffs, d_coeffs)
        

        error = calculate_error(u_next, x_points, t, a, N)
        errors.append((t, error))
        solutions.append((t, u_next.copy()))

        u_prev, u_curr = u_curr, u_next
        u_next = [0.0] * (N + 1)
    
    return errors, solutions

def plot_all_schemes_comparison():
    L = math.pi
    T = 1.0
    K = 2000
    N = 1000
    a = 1.0
    
    schemes_to_test = [
        ('explicit', 'two_point_first', 'first_order', 'Явная: 2-точ.гран., 1-пор.нач.'),
        ('explicit', 'two_point_first', 'second_order', 'Явная: 2-точ.гран., 2-пор.нач.'),
        ('explicit', 'three_point_second', 'first_order', 'Явная: 3-точ.гран., 1-пор.нач.'),
        ('explicit', 'three_point_second', 'second_order', 'Явная: 3-точ.гран., 2-пор.нач.'),
        ('explicit', 'two_point_second', 'first_order', 'Явная: 2-точ.гран.2, 1-пор.нач.'),
        ('explicit', 'two_point_second', 'second_order', 'Явная: 2-точ.гран.2, 2-пор.нач.'),

        ('implicit', 'two_point_first', 'first_order', 'Неявная: 2-точ.гран., 1-пор.нач.'),
        ('implicit', 'two_point_first', 'second_order', 'Неявная: 2-точ.гран., 2-пор.нач.'),
        ('implicit', 'three_point_second', 'first_order', 'Неявная: 3-точ.гран., 1-пор.нач.'),
        ('implicit', 'three_point_second', 'second_order', 'Неявная: 3-точ.гран., 2-пор.нач.'),
        ('implicit', 'two_point_second', 'first_order', 'Неявная: 2-точ.гран.2, 1-пор.нач.'),
        ('implicit', 'two_point_second', 'second_order', 'Неявная: 2-точ.гран.2, 2-пор.нач.'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    x_analytic = np.linspace(0, L, 200)

    colors = plt.cm.tab20(np.linspace(0, 1, 12))
    
    for idx, (scheme_type, bc_type, init_approx, scheme_name) in enumerate(schemes_to_test):
        
        try:
            if scheme_type == 'explicit':
                errors, solutions = explicit_scheme(L, N, T, K, a, bc_type, init_approx)
            else:
                errors, solutions = implicit_scheme(L, N, T, K, a, bc_type, init_approx)
            
            x_points = [i * L / N for i in range(N + 1)]
            
            mid_idx = len(solutions) // 2
            end_idx = len(solutions) - 1
            
            if scheme_type == 'explicit':
                if len(solutions) >= 2:
                    # t = 0.5
                    axes[0, 0].plot(x_points, solutions[mid_idx][1], '--', color=colors[idx], 
                                   label=scheme_name, linewidth=2, alpha=0.8)
                    # t = 1.0
                    axes[0, 1].plot(x_points, solutions[end_idx][1], '--', color=colors[idx], 
                                   label=scheme_name, linewidth=2, alpha=0.8)
            else:
                if len(solutions) >= 2:
                    # t = 0.5
                    axes[1, 0].plot(x_points, solutions[mid_idx][1], '--', color=colors[idx], 
                                   label=scheme_name, linewidth=2, alpha=0.8)
                    # t = 1.0
                    axes[1, 1].plot(x_points, solutions[end_idx][1], '--', color=colors[idx], 
                                   label=scheme_name, linewidth=2, alpha=0.8)
        except Exception as e:
            print(f"Ошибка при построении {scheme_name}: {e}")

    analytic_times = [0.5, 1.0]
    for i, t_val in enumerate(analytic_times):
        axes[0, i].plot(x_analytic, U(x_analytic, t_val, a), 'k-', label='Аналитическое', linewidth=2)
        axes[1, i].plot(x_analytic, U(x_analytic, t_val, a), 'k-', label='Аналитическое', linewidth=2)
    
    titles = ['Явная схема, t=0.5', 'Явная схема, t=1.0', 
              'Неявная схема, t=0.5', 'Неявная схема, t=1.0']
    
    for ax, title in zip(axes.flat, titles):
        ax.set_xlabel('x')
        ax.set_ylabel('u(x,t)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, loc='upper right')
    
    plt.tight_layout()
    plt.show()

def plot_all_errors_comparison():
    L = math.pi
    T = 1.0
    K = 200
    N = 100
    a = 1.0
    
    schemes_to_test = [
        ('explicit', 'two_point_first', 'first_order', 'Явная: 2-точ.гран., 1-пор.нач.'),
        ('explicit', 'two_point_first', 'second_order', 'Явная: 2-точ.гран., 2-пор.нач.'),
        ('explicit', 'three_point_second', 'first_order', 'Явная: 3-точ.гран., 1-пор.нач.'),
        ('explicit', 'three_point_second', 'second_order', 'Явная: 3-точ.гран., 2-пор.нач.'),
        ('explicit', 'two_point_second', 'first_order', 'Явная: 2-точ.гран.2, 1-пор.нач.'),
        ('explicit', 'two_point_second', 'second_order', 'Явная: 2-точ.гран.2, 2-пор.нач.'),

        ('implicit', 'two_point_first', 'first_order', 'Неявная: 2-точ.гран., 1-пор.нач.'),
        ('implicit', 'two_point_first', 'second_order', 'Неявная: 2-точ.гран., 2-пор.нач.'),
        ('implicit', 'three_point_second', 'first_order', 'Неявная: 3-точ.гран., 1-пор.нач.'),
        ('implicit', 'three_point_second', 'second_order', 'Неявная: 3-точ.гран., 2-пор.нач.'),
        ('implicit', 'two_point_second', 'first_order', 'Неявная: 2-точ.гран.2, 1-пор.нач.'),
        ('implicit', 'two_point_second', 'second_order', 'Неявная: 2-точ.гран.2, 2-пор.нач.'),
    ]
    
    plt.figure(figsize=(14, 8))
    
    colors = plt.cm.tab20(np.linspace(0, 1, 12))
    
    all_errors_data = []
    
    for idx, (scheme_type, bc_type, init_approx, scheme_name) in enumerate(schemes_to_test):
        
        if scheme_type == 'explicit':
            errors, _ = explicit_scheme(L, N, T, K, a, bc_type, init_approx)
        else:
            errors, _ = implicit_scheme(L, N, T, K, a, bc_type, init_approx)
        
        times = [err[0] for err in errors]
        error_vals = [err[1] for err in errors]
        
        all_errors_data.append((scheme_name, times, error_vals))
        
        linestyle = '-' 
        if scheme_type == 'explicit': 
            linestyle = '--'
        plt.semilogy(times, error_vals, linestyle, color=colors[idx], 
                    label=scheme_name, linewidth=2, alpha=0.8)
    
    plt.xlabel('t')
    plt.ylabel('Норма ошибки')
    plt.title('Динамика ошибок')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
    return all_errors_data

def plot_grid_dependence_all():
    L = math.pi
    T = 1.0
    a = 1.0

    N_values = [20, 30, 40, 50, 60, 70, 80]
    K_values_space = [int(2 * N) for N in N_values] 

    K_values_time = [50, 100, 200, 300, 400, 500, 600]
    N_values_time = [50] * len(K_values_time) 
    
    schemes_to_test = [
        ('explicit', 'two_point_first', 'first_order', 'Явная: 2-точ.гран., 1-пор.нач.'),
        ('explicit', 'two_point_first', 'second_order', 'Явная: 2-точ.гран., 2-пор.нач.'),
        ('explicit', 'three_point_second', 'first_order', 'Явная: 3-точ.гран., 1-пор.нач.'),
        ('explicit', 'three_point_second', 'second_order', 'Явная: 3-точ.гран., 2-пор.нач.'),
        ('explicit', 'two_point_second', 'first_order', 'Явная: 2-точ.гран.2, 1-пор.нач.'),
        ('explicit', 'two_point_second', 'second_order', 'Явная: 2-точ.гран.2, 2-пор.нач.'),

        ('implicit', 'two_point_first', 'first_order', 'Неявная: 2-точ.гран., 1-пор.нач.'),
        ('implicit', 'two_point_first', 'second_order', 'Неявная: 2-точ.гран., 2-пор.нач.'),
        ('implicit', 'three_point_second', 'first_order', 'Неявная: 3-точ.гран., 1-пор.нач.'),
        ('implicit', 'three_point_second', 'second_order', 'Неявная: 3-точ.гран., 2-пор.нач.'),
        ('implicit', 'two_point_second', 'first_order', 'Неявная: 2-точ.гран.2, 1-пор.нач.'),
        ('implicit', 'two_point_second', 'second_order', 'Неявная: 2-точ.гран.2, 2-пор.нач.'),
    ]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(schemes_to_test)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+']
    
    # График 1: Зависимость от шага по пространству h
    for idx, (scheme_type, bc_type, init_approx, scheme_name) in enumerate(schemes_to_test):
        final_errors = []
        
        for N, K in zip(N_values, K_values_space):
            
            try:
                if scheme_type == 'explicit':
                    errors, _ = explicit_scheme(L, N, T, K, a, bc_type, init_approx)
                else:
                    errors, _ = implicit_scheme(L, N, T, K, a, bc_type, init_approx)
                
                if errors:
                    final_errors.append(errors[-1][1])
                else:
                    final_errors.append(0.0)
            except Exception as e:
                print(f"Ошибка при тестировании {scheme_name} с N={N}: {e}")
                final_errors.append(np.nan)
        
        h_values = [L / N for N in N_values]
        valid_indices = [i for i, err in enumerate(final_errors) if not np.isnan(err)]
        if valid_indices:
            valid_h = [h_values[i] for i in valid_indices]
            valid_errors = [final_errors[i] for i in valid_indices]
            ax1.loglog(valid_h, valid_errors, marker=markers[idx], color=colors[idx], 
                      label=scheme_name, linewidth=2, markersize=6, alpha=0.8)
    
    ax1.set_xlabel('h')
    ax1.set_ylabel('Jшибка')
    ax1.set_title('Зависимость погрешности от шага по пространству')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # График 2: Зависимость от шага по времени tao
    for idx, (scheme_type, bc_type, init_approx, scheme_name) in enumerate(schemes_to_test):
        final_errors = []
        
        for N, K in zip(N_values_time, K_values_time):
            
            try:
                if scheme_type == 'explicit':
                    errors, _ = explicit_scheme(L, N, T, K, a, bc_type, init_approx)
                else:
                    errors, _ = implicit_scheme(L, N, T, K, a, bc_type, init_approx)
                
                if errors:
                    final_errors.append(errors[-1][1])
                else:
                    final_errors.append(0.0)
            except Exception as e:
                print(f"Ошибка при тестировании {scheme_name} с K={K}: {e}")
                final_errors.append(np.nan)
        
        tao_values = [T / K for K in K_values_time]
        valid_indices = [i for i, err in enumerate(final_errors) if not np.isnan(err)]
        if valid_indices:
            valid_tao = [tao_values[i] for i in valid_indices]
            valid_errors = [final_errors[i] for i in valid_indices]
            ax2.loglog(valid_tao, valid_errors, marker=markers[idx], color=colors[idx], 
                      label=scheme_name, linewidth=2, markersize=6, alpha=0.8)
    
    ax2.set_xlabel('tao')
    ax2.set_ylabel('Ошибка')
    ax2.set_title('Зависимость погрешности от шага по времени')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.show()

def main():
    L = math.pi
    T = 1.0
    K = 50
    N = 50
    a = 1.0

    schemes_to_test = [
        ('explicit', 'two_point_first', 'first_order', 'Явная: 2-точ.гран., 1-пор.нач.'),
        ('explicit', 'two_point_first', 'second_order', 'Явная: 2-точ.гран., 2-пор.нач.'),
        ('explicit', 'three_point_second', 'first_order', 'Явная: 3-точ.гран., 1-пор.нач.'),
        ('explicit', 'three_point_second', 'second_order', 'Явная: 3-точ.гран., 2-пор.нач.'),
        ('explicit', 'two_point_second', 'first_order', 'Явная: 2-точ.гран.2, 1-пор.нач.'),
        ('explicit', 'two_point_second', 'second_order', 'Явная: 2-точ.гран.2, 2-пор.нач.'),
        
        ('implicit', 'two_point_first', 'first_order', 'Неявная: 2-точ.гран., 1-пор.нач.'),
        ('implicit', 'two_point_first', 'second_order', 'Неявная: 2-точ.гран., 2-пор.нач.'),
        ('implicit', 'three_point_second', 'first_order', 'Неявная: 3-точ.гран., 1-пор.нач.'),
        ('implicit', 'three_point_second', 'second_order', 'Неявная: 3-точ.гран., 2-пор.нач.'),
        ('implicit', 'two_point_second', 'first_order', 'Неявная: 2-точ.гран.2, 1-пор.нач.'),
        ('implicit', 'two_point_second', 'second_order', 'Неявная: 2-точ.гран.2, 2-пор.нач.'),
    ]
    
    results_explicit = {}
    results_implicit = {}
    
    for scheme_type, bc_type, init_approx, scheme_name in schemes_to_test:
        
        if scheme_type == 'explicit':
            errors, solutions = explicit_scheme(L, N, T, K, a, bc_type, init_approx)
            results_explicit[scheme_name] = (errors, solutions)
        else:
            errors, solutions = implicit_scheme(L, N, T, K, a, bc_type, init_approx)
            results_implicit[scheme_name] = (errors, solutions)

    plot_all_schemes_comparison()
    plot_all_errors_comparison()
    plot_grid_dependence_all()
    
    print("\n" + "="*80)
    print("ИТОГОВАЯ ТАБЛИЦА ОШИБОК:")
    print("="*80)
    print("\nЯВНЫЕ СХЕМЫ:")
    for scheme_name, (errors, _) in results_explicit.items():
        if errors:
            final_error = errors[-1][1]
            print(f"{scheme_name:50} - конечная ошибка = {final_error:.2e}")
    
    print("\nНЕЯВНЫЕ СХЕМЫ:")
    for scheme_name, (errors, _) in results_implicit.items():
        if errors:
            final_error = errors[-1][1]
            print(f"{scheme_name:50} - конечная ошибка = {final_error:.2e}")

if __name__ == "__main__":
    main()