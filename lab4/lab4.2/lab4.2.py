import math
import matplotlib.pyplot as plt

def read_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        dy0 = float(lines[0].strip())
        dy1_y1 = float(lines[1].strip())
    return dy0, dy1_y1

def write_results(filename, dy0, dy1_y1, x_shooting, y_shooting, z_shooting, exact_x, exact_y, x_finite_diff, y_finite_diff, 
                 shooting_errors_rr, shooting_errors_exact, iter, fd_errors_rr, fd_errors_exact):
    with open(filename, "w", encoding='utf-8') as file:
        file.write(f"Краевая задача\n")
        file.write(f"(e^x + 1)y'' - 2y' - e^x * y = 0\n")
        file.write(f"y'(0) = {dy0}\n")
        file.write(f"y'(1) - y(1) = {dy1_y1}\n")

        file.write("\nМетод стрельбы:\n")
        file.write(f"\nИтерации: {iter}\n")
        file.write("x\t\tПриближенное y\t Приближенное y'\t Погрешность Рунге\tТочная погрешность\n")
        for i in range(len(x_shooting)):
            file.write(f"{x_shooting[i]:.4f}\t{y_shooting[i]:.10f}\t{z_shooting[i]:.10f}\t{shooting_errors_rr[i]:.10f}\t{shooting_errors_exact[i]:.10f}\n")

        file.write("\nКонечно-разностный метод:\n")
        file.write("x\t\tПриближенное y\tПогрешность Рунге\tТочная погрешность\n")
        for i in range(len(x_finite_diff)):
            file.write(f"{x_finite_diff[i]:.4f}\t{y_finite_diff[i]:.10f}\t{fd_errors_rr[i]:.10f}\t{fd_errors_exact[i]:.10f}\n")

        file.write("\nТочное решение:\n")
        file.write("x\t\tТочное y\n")
        for i in range(len(exact_x)):
            file.write(f"{exact_x[i]:.4f}\t{exact_y[i]:.10f}\n")
            
        avg_shooting_rr = sum(shooting_errors_rr) / len(shooting_errors_rr)
        avg_shooting_exact = sum(shooting_errors_exact) / len(shooting_errors_exact)
        avg_fd_rr = sum(fd_errors_rr) / len(fd_errors_rr)
        avg_fd_exact = sum(fd_errors_exact) / len(fd_errors_exact)
        
        file.write("\nСредние погрешности:\n")
        file.write("Метод\t\tСредняя погрешность Рунге\tСредняя точная погрешность\n")
        file.write(f"Стрельбы\t{avg_shooting_rr:.10f}\t\t\t{avg_shooting_exact:.10f}\n")
        file.write(f"Конечно-разн.\t{avg_fd_rr:.10f}\t\t\t{avg_fd_exact:.10f}\n")

def f(x, y, z):
    """
    (e^x + 1)y'' - 2y' - e^x * y = 0
    y' = z
    z' = (2 * z + y * e^x) / (e^x + 1)
    """
    return (2 * z + y * math.exp(x)) / (math.exp(x) + 1)

def g(x, y, z):
    return z

def p(x):
    return -2 / (math.exp(x) + 1)

def q(x):
    return -math.exp(x) / (math.exp(x) + 1)

def df(x):
    return 0

def exact_solution(x):
    return math.exp(x) - 1

def runge_kutta_method(y0, dy0, a, b, h):
    x_val = []
    y_val = []
    z_val = []

    x = a
    y = y0
    z = dy0

    while x <= b + h/2:
        x_val.append(x)
        y_val.append(y)
        z_val.append(z)

        k1 = h * g(x, y, z)
        l1 = h * f(x, y, z)

        k2 = h * g(x + h/2, y + k1/2, z + l1/2)
        l2 = h * f(x + h/2, y + k1/2, z + l1/2)
        
        k3 = h * g(x + h/2, y + k2/2, z + l2/2)
        l3 = h * f(x + h/2, y + k2/2, z + l2/2)
        
        k4 = h * g(x + h, y + k3, z + l3)
        l4 = h * f(x + h, y + k3, z + l3)
        
        y += (k1 + 2*k2 + 2*k3 + k4) / 6
        z += (l1 + 2*l2 + 2*l3 + l4) / 6
        x += h
    return x_val, y_val, z_val

def shooting_method(dy0, dy1_y1, a, b, h, eps=1e-6, max_iter=100):
    """
    Метод стрельбы для решения краевой задачи:
    1. Преобразуем краевую задачу к задаче Коши с параметром alpha (начальное значение y(0))
    2. Используем итерационный метод (Ньютона или секущих) для подбора alpha:
       - На каждой итерации решаем задачу Коши методом Рунге-Кутты 4 порядка
       - Проверяем выполнение правого граничного условия
    3. Итерации продолжаются, пока не достигнута заданная точность или max_iter
    4. Возвращаем решение и найденные y(0) и y(1)
    """
    alpha_prev = 0.0
    alpha_curr = 10.0
    iter = 0
    for _ in range(max_iter):
        iter += 1
        x, y, z = runge_kutta_method(alpha_curr, dy0, a, b, h)
        residual = z[-1] - y[-1] - dy1_y1
        
        if abs(residual) < eps:
            break
            
        x_prev, y_prev, z_prev = runge_kutta_method(alpha_prev, dy0, a, b, h)
        residual_prev = z_prev[-1] - y_prev[-1] - dy1_y1
        
        if residual == residual_prev:
            break
            
        alpha_next = alpha_curr - residual * (alpha_curr - alpha_prev) / (residual - residual_prev)
        
        alpha_prev, alpha_curr = alpha_curr, alpha_next
    
    return x, y, z, y[0], y[-1], iter

def tridiagonal_matrix_algorithm(A, b):
    n = len(A)
    if n == 0:
        return []
    
    P = [0.0 for _ in range(n)]
    Q = [0.0 for _ in range(n)]
    
    P[0] = -A[0][1] / A[0][0]
    Q[0] = b[0] / A[0][0]
    
    for i in range(1, n-1):
        denominator = A[i][i] + A[i][i-1] * P[i-1]
        P[i] = -A[i][i+1] / denominator
        Q[i] = (b[i] - A[i][i-1] * Q[i-1]) / denominator
    
    y = [0.0 for _ in range(n)]
    y[-1] = (b[-1] - A[-1][-2] * Q[-2]) / (A[-1][-1] + A[-1][-2] * P[-2])
    
    for i in range(n-2, -1, -1):
        y[i] = P[i] * y[i+1] + Q[i]
    
    return y

def finite_difference_method(y0, y1, a, b, h):
    """
    1. Строим равномерную сетку на отрезке [a, b] с шагом h
    2. Аппроксимируем производные конечными разностями:
       - y'' ≈ (y_{i-1} - 2y_i + y_{i+1})/h²
       - y' ≈ (y_{i+1} - y_{i-1})/(2h)
    3. Для каждого внутреннего узла записываем разностное уравнение
    4. Граничные условия задаются явно: y(a) = y0, y(b) = y1
    5. Полученную трехдиагональную систему решаем методом прогонки
    """
    n = int((b - a) / h)
    x = [a + i * h for i in range(n + 1)]
    
    A = [[0.0 for _ in range(n + 1)] for _ in range(n + 1)]
    B = [0.0 for _ in range(n + 1)]
    
    A[0][0] = 1
    B[0] = y0
    
    for i in range(1, n):
        xi = x[i]
        A[i][i-1] = 1 / h**2 - p(xi) / (2 * h)
        A[i][i] = -2 / h**2 + q(xi)
        A[i][i+1] = 1 / h**2 + p(xi) / (2 * h)
        B[i] = df(xi)
    
    A[n][n] = 1
    B[n] = y1
    
    y = tridiagonal_matrix_algorithm(A, B)
    
    return x, y


def runge_romberg_method(h1, y_h1, h2, y_h2, p):
    error = []
    for i in range(len(y_h1)):
        err = abs(y_h1[i] - y_h2[2*i]) / (2**p - 1)
        error.append(err)
    return error

def calculate_exact_errors(y_approx, exact_y):
    return [abs(y_approx[i] - exact_y[i]) for i in range(len(y_approx))]

def plot_results(x_shooting, y_shooting, exact_x, exact_y, x_finite_diff, y_finite_diff, shooting_errors, fd_errors, shooting_errors_rr, fd_errors_rr):
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 1, 1)
    plt.plot(x_shooting, y_shooting, 'b-', label='Метод стрельбы')
    plt.plot(x_finite_diff, y_finite_diff, 'g-', label='Конечно-разностный метод')
    plt.plot(exact_x, exact_y, 'k--', label='Точное решение')
    plt.xlabel('x')
    plt.ylabel('y(x)')
    plt.title('Сравнение методов решения краевой задачи')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(x_shooting, shooting_errors, 'b-', label='Погрешность метода стрельбы')
    plt.plot(x_shooting, shooting_errors_rr, 'r-', label='Погрешность метода стрельбы rr')
    plt.plot(x_finite_diff, fd_errors, 'g-', label='Погрешность конечно-разностного метода')
    plt.plot(x_finite_diff, fd_errors_rr, 'c-', label='Погрешность конечно-разностного метода rr')
    plt.xlabel('x')
    plt.ylabel('Погрешность')
    plt.title('Сравнение погрешностей')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def main():
    dy0, dy1_y1 = read_file("input.txt")
    a, b = 0.0, 1.0
    h = 0.1

    x_shooting, y_shooting, z_shooting, y0, y1, iter = shooting_method(dy0, dy1_y1, a, b, h)
    
    x_finite_diff, y_finite_diff = finite_difference_method(y0, y1, a, b, h)

    exact_x = [a + i * h for i in range(int((b - a) / h) + 1)]
    exact_y = [exact_solution(x) for x in exact_x]

    h2 = h / 2
    x_shooting_h2, y_shooting_h2, _,  y0_h2, y1_h2, _ = shooting_method(dy0, dy1_y1, a, b, h2)
    x_fd_h2, y_fd_h2 = finite_difference_method(y0_h2, y1_h2, a, b, h2)

    shooting_errors_rr = runge_romberg_method(h, y_shooting, h2, y_shooting_h2, 4)
    fd_errors_rr = runge_romberg_method(h, y_finite_diff, h2, y_fd_h2, 2)
    
    shooting_errors_exact = calculate_exact_errors(y_shooting, exact_y)
    fd_errors_exact = calculate_exact_errors(y_finite_diff, exact_y)

    plot_results(x_shooting, y_shooting, exact_x, exact_y, x_finite_diff, y_finite_diff, 
                shooting_errors_exact, fd_errors_exact, shooting_errors_rr, fd_errors_rr)

    write_results("output.txt", dy0, dy1_y1, x_shooting, y_shooting, z_shooting, exact_x, exact_y, 
                 x_finite_diff, y_finite_diff, shooting_errors_rr, shooting_errors_exact, iter,
                 fd_errors_rr, fd_errors_exact)

if __name__ == "__main__":
    main()