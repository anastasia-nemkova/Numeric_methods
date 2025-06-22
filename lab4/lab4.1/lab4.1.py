import math
import matplotlib.pyplot as plt

def read_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        y0 = float(lines[0].strip())
        dy0 = float(lines[1].strip())
        a, b = map(float, lines[2].strip().split())
        h = float(lines[3].strip())
    return y0, dy0, a, b, h

def write_results(filename, y0, dy0, a, b, h, x_euler, y_euler, x_rk, y_rk, exact_x, exact_y, x_adams, y_adams, error_euler, error_rk, error_adams, exact_error_euler, exact_error_rk, exact_error_adams):
    with open(filename, "w", encoding='utf-8') as file:
        file.write(f"Задача Коши\n")
        file.write(f"y'' + 2y'*ctgx + 3y = 0\n")
        file.write(f"y(0) = {y0}\n")
        file.write(f"y'(0) = {dy0}\n")
        file.write(f"x ∈ [{a}, {b}],  h = {h}\n") 

        file.write(f"\nМетод Эйлера\n")
        file.write("x\t\tПриближенное y\tПогрешность Рунге\tТочная погрешность\n")
        for i in range(len(x_euler)):
            file.write(f"{x_euler[i]:.4f}\t\t{y_euler[i]:.6f}\t\t{error_euler[i]:.6f}\t\t{exact_error_euler[i]:.6f}\n")

        file.write("\nМетод Рунге-Кутты 4-го порядка:\n")
        file.write("x\t\tПриближенное y\tПогрешность Рунге\tТочная погрешность\n")
        for i in range(len(x_rk)):
            file.write(f"{x_rk[i]:.4f}\t\t{y_rk[i]:.6f}\t\t{error_rk[i]:.6f}\t\t{exact_error_rk[i]:.6f}\n")

        file.write("\nМетод Адамса 4-го порядка:\n")
        file.write("x\t\tПриближенное y\tПогрешность Рунге\tТочная погрешность\n")
        for i in range(len(x_adams)):
            file.write(f"{x_adams[i]:.4f}\t\t{y_adams[i]:.6f}\t\t{error_adams[i]:.6f}\t\t{exact_error_adams[i]:.6f}\n")

        file.write("\nТочное решение:\n")
        file.write("x\t\tТочное y\n")
        for i in range(len(exact_x)):
            file.write(f"{exact_x[i]:.4f}\t{exact_y[i]:.6f}\n")

        file.write("\nОбщие результаты:\n")
        avg_euler = sum(error_euler)/len(error_euler)
        avg_euer_exact = sum(exact_error_euler)/len(exact_error_euler)
        avg_rk = sum(error_rk)/len(error_rk)
        avg_rk_exact = sum(exact_error_rk)/len(exact_error_rk)
        avg_adams = sum(error_adams)/len(error_adams)
        avg_adams_exact = sum(exact_error_adams)/len(exact_error_adams)
        file.write(f"Метод Эйлера: погрешность = {avg_euler:.6f}, точная погрешность = {avg_euer_exact:.6f}\n")
        file.write(f"Метод Рунге-Кутты: погрешность = {avg_rk:.6f}, точная погрешность = {avg_rk_exact:.6f}\n")
        file.write(f"Метод Адамса: погрешность = {avg_adams:.6f}, точная погрешность = {avg_adams_exact:.6f}\n")

def f(x, y, z):
    '''
    y'' + 2y'*ctgx + 3y = 0
    y' = z
    z' = - 2 * z * ctgx - 3y
    '''
    return - 2 * (math.cos(x) / math.sin(x)) * z - 3 * y

def g(x, y, z):
    return z

def exact_solution(x):
    return (-0.9783 * math.cos(2 * x) + 0.4776 * math.sin(2 * x)) / math.sin(x)

def euler_method(y0, dy0, a, b, h):
    """
    Метод Эйлера для решения ОДУ 2-го порядка.
    Разбиваем уравнение на систему двух уравнений 1-го порядка:
    y' = z = g(x, y, z)
    z' = f(x, y, z)
    На каждом шаге вычисляем новые значения y и z по формулам:
    y_{n+1} = y_n + h * g(x_n, y_n, z_n)
    z_{n+1} = z_n + h * f(x_n, y_n, z_n)
    """
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

        z += h * f(x, y , z)
        y_new = y + h * g(x, y, z)
        

        y= y_new
        x += h

    return x_val, y_val

def runge_kutta_method(y0, dy0, a, b, h, return_z=False):
    """
    Метод Рунге-Кутты 4-го порядка для решения ОДУ 2-го порядка.
    Аналогично разбиваем на систему двух уравнений.
    Для каждого шага вычисляем 4 коэффициента (k1-k4 для y, l1-l4 для z):
    k1 = h * g(x_n, y_n, z_n)
    l1 = h * f(x_n, y_n, z_n)
    
    k2 = h * g(x_n + h/2, y_n + k1/2, z_n + l1/2)
    l2 = h * f(x_n + h/2, y_n + k1/2, z_n + l1/2)
    
    k3 = h * g(x_n + h/2, y_n + k2/2, z_n + l2/2)
    l3 = h * f(x_n + h/2, y_n + k2/2, z_n + l2/2)
    
    k4 = h * g(x_n + h, y_n + k3, z_n + l3)
    l4 = h * f(x_n + h, y_n + k3, z_n + l3)
    
    Затем обновляем значения:
    y_{n+1} = y_n + (k1 + 2k2 + 2k3 + k4)/6
    z_{n+1} = z_n + (l1 + 2l2 + 2l3 + l4)/6
    """
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
    if return_z:
        return x_val, y_val, z_val
    else:
        return x_val, y_val

def adams_method(y0, dy0, a, b, h):
    """
    Метод Адамса 4-го порядка для решения ОДУ 2-го порядка.
    Использует 4 предыдущих значения для вычисления следующего.
    Сначала нужно получить 4 начальных значения методом Рунге-Кутты.
    Затем для каждого следующего шага:
    y_{n+1} = y_n + h*(55g_n - 59g_{n-1} + 37g_{n-2} - 9g_{n-3})/24
    z_{n+1} = z_n + h*(55f_n - 59f_{n-1} + 37f_{n-2} - 9f_{n-3})/24
    где g_n = g(x_n, y_n, z_n), f_n = f(x_n, y_n, z_n)
    """
    x_rk, y_rk, z_rk = runge_kutta_method(y0, dy0, a, b, h, return_z=True)
    x_val = x_rk[:4]
    y_val = y_rk[:4]
    z_val = z_rk[:4]

    g_vals = [g(x_val[i], y_val[i], z_val[i]) for i in range(4)]
    f_vals = [f(x_val[i], y_val[i], z_val[i]) for i in range(4)]

    x = x_val[-1] + h
    while x <= b + h/2:
        y_new = y_val[-1] + h*(55*g_vals[-1] - 59*g_vals[-2] + 37*g_vals[-3] - 9*g_vals[-4])/24
        z_new = z_val[-1] + h*(55*f_vals[-1] - 59*f_vals[-2] + 37*f_vals[-3] - 9*f_vals[-4])/24

        x_val.append(x)
        y_val.append(y_new)
        z_val.append(z_new)

        g_new = g(x, y_new, z_new)
        f_new = f(x, y_new, z_new)
        
        g_vals.pop(0)
        g_vals.append(g_new)
        f_vals.pop(0)
        f_vals.append(f_new)
        
        x += h
    
    return x_val, y_val

def runge_romberg_method(h1, y_h1, h2, y_h2, p):
    error = []
    for i in range(len(y_h1)):
        err = abs(y_h1[i] - y_h2[2*i]) / (2**p - 1)
        error.append(err)
    return error

def plot_results(x_euler, y_euler, x_rk, y_rk, exact_x, exact_y, x_adams, y_adams, error_euler, error_rk, error_adams, exact_error_euler, exact_error_rk, exact_error_adams):
    plt.figure(figsize=(15, 10))
    
    # Графики решений
    plt.subplot(3, 1, 1)
    plt.plot(x_euler, y_euler, 'b-', label='Метод Эйлера')
    plt.plot(x_rk, y_rk, 'g-', label='Метод Рунге-Кутты 4-го порядка')
    plt.plot(x_adams, y_adams, 'r-', label='Метод Адамса 4-го порядка')
    plt.plot(exact_x, exact_y, 'k--', label='Точное решение')
    plt.xlabel('x')
    plt.ylabel('y(x)')
    plt.title('Сравнение численных методов')
    plt.legend()
    plt.grid(True)
    
    # Графики погрешностей по Рунге-Ромбергу
    plt.subplot(3, 1, 2)
    plt.plot(x_euler, error_euler, 'b-', label='Погрешность Эйлера (Рунге-Ромберг)')
    plt.plot(x_rk, error_rk, 'g-', label='Погрешность Рунге-Кутты (Рунге-Ромберг)')
    plt.plot(x_adams, error_adams, 'r-', label='Погрешность Адамса (Рунге-Ромберг)')
    plt.xlabel('x')
    plt.ylabel('Погрешность')
    plt.title('Оценка погрешностей методом Рунге-Ромберга')
    plt.legend()
    plt.grid(True)
    
    # Графики точных погрешностей
    plt.subplot(3, 1, 3)
    plt.plot(x_euler, exact_error_euler, 'b-', label='Точная погрешность Эйлера')
    plt.plot(x_rk, exact_error_rk, 'g-', label='Точная погрешность Рунге-Кутты')
    plt.plot(x_adams, exact_error_adams, 'r-', label='Точная погрешность Адамса')
    plt.xlabel('x')
    plt.ylabel('Погрешность')
    plt.title('Сравнение с точным решением')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    y0, dy0, a, b, h = read_file("input.txt")

    x_euler, y_euler = euler_method(y0, dy0, a, b, h)
    x_rk, y_rk = runge_kutta_method(y0, dy0, a, b, h)
    x_adams, y_adams = adams_method(y0, dy0, a, b, h)

    exact_x = [a + i*h for i in range(int((b-a)*10) + 1)]
    exact_y = [exact_solution(x) for x in exact_x]

    _, y_euler_h2 = euler_method(y0, dy0, a, b, h/2)
    _, y_rk_h2 = runge_kutta_method(y0, dy0, a, b, h/2)
    _, y_adams_h2 = adams_method(y0, dy0, a, b, h/2)
    
    # Для методов Эйлера (порядок 1), Рунге-Кутты (порядок 4), Адамса (порядок 4)
    error_euler = runge_romberg_method(h, y_euler, h/2, y_euler_h2, 1)
    error_rk = runge_romberg_method(h, y_rk, h/2, y_rk_h2, 4)
    error_adams = runge_romberg_method(h, y_adams, h/2, y_adams_h2, 4)

    exact_error_euler = [abs(y_euler[i] - exact_solution(x_euler[i])) for i in range(len(x_euler))]
    exact_error_rk = [abs(y_rk[i] - exact_solution(x_rk[i])) for i in range(len(x_rk))]
    exact_error_adams = [abs(y_adams[i] - exact_solution(x_adams[i])) for i in range(len(x_adams))]

    plot_results(x_euler, y_euler, x_rk, y_rk, exact_x, exact_y, x_adams, y_adams, error_euler, error_rk, error_adams, exact_error_euler, exact_error_rk, exact_error_adams)
    write_results("output.txt", y0, dy0, a, b, h, x_euler, y_euler, x_rk, y_rk, exact_x, exact_y, x_adams, y_adams, error_euler, error_rk, error_adams, exact_error_euler, exact_error_rk, exact_error_adams)


if __name__ == "__main__":
    main()