import math
import matplotlib.pyplot as plt
import numpy as np

# Функция уравнения
def f(x):
    return 3 ** x - 5 * x ** 2 + 1

# Производная f(x)
def df(x):
    return math.log(3)*3 ** x - 10 * x

# Вторая производная f(x)
def ddf(x):
    return math.log(3) ** 2 * 3 ** x - 10

def fi(x):
    return math.sqrt((3 ** x + 1) / 5)

def dfi(x):
    return (math.sqrt(5) * math.log(3) * 3 ** x) / (10 * math.sqrt(3 ** x + 1))

# q = max_[a, b] |fi'(x)|
def get_q(a, b):
    return max(abs(dfi(a)), abs(dfi(b)))

# m1 = min_[a, b] |f'(x)|
def get_m1(a, b):
    return min(abs(df(a)), abs(df(b)))

# M2 = max_[a, b] |f''(x)|
def get_M2(a, b):
    return max(abs(ddf(a)), abs(ddf(b)))

# Проверка условия f(a) * f(b) < 0
def check_condition(a, b):
    if f(a) * f(b) >= 0:
        return False
    return True

def iteration_method(epsilon, a, b, max_iter = 100):
    '''
    x_k = fi(x_(k - 1))
    eps_k = q / (1 - q) * |x_k - x_(k - 1)|
    eps_k <= eps => finish
    '''
    
    x0 = (a + b) / 2
    q = get_q(a, b)
    iter = 0
    
    errors = []
    
    while iter < max_iter:
        iter += 1
        x = fi(x0)
        eps_now = q / (1 - q) * abs(x - x0)
        errors.append(eps_now)
        
        if eps_now <= epsilon:
            break
        x0 = x
    return x , iter, errors

def Newton_method(epsilon, a, b, max_iter = 100):
    '''
    x_k = x_(k - 1) - f(x_(k - 1)) / f'(x_(k - 1))
    eps_ k = (M2 / (2 * m1)) * (x_(k - 1) - x)^2
    eps_k <= epsilon => finish
    '''
    
    x0 = (a + b) / 2
    iter = 0
    m1 = get_m1(a, b)
    M2 = get_M2(a, b)
    
    errors = []
    
    while iter < max_iter:
        iter += 1
        x = x0 - f(x0) / df(x0)
        
        eps_now = M2 / (2 * m1) * (x0 - x) ** 2
        errors.append(eps_now)
        
        if eps_now <= epsilon:
            break
        x0 = x
    return x, iter, errors
    
def read_file(filename):
    with open(filename, 'r',) as file:
        epsilon = float(file.readline().strip())
        a, b = map(float, file.readline().strip().split())
    return epsilon, a, b
    
    
def write_in_file(filename, a, b, epsilon, root_iter, iterations_iter, root_Newt, iterations_Newt):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(f"Уравнение: 3^x - 5*x^2 + 1 = 0\n")
        file.write(f"Отрезок: [{a}, {b}]\n")
        file.write(f"Точность: {epsilon}\n")
        
        file.write(f"\nМетод простой итерации\n")
        file.write(f"Корень: {root_iter}\n")
        file.write(f"Количество итераций: {iterations_iter}\n")
        file.write(f"f(x) = {f(root_iter)}\n")
        
        file.write(f"\nМетод Ньютона\n")
        file.write(f"Корень: {root_Newt}\n")
        file.write(f"Количество итераций: {iterations_Newt}\n")
        file.write(f"f(x) = {f(root_Newt)}\n")

def write_error_in_file(filename, message):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(f"Ошибка: {message}\n")


def plot_all(errors_iter, errors_Newt):
    x_values = np.linspace(0, 2, 400)
    y_values = f(x_values)

    plt.figure(figsize=(12, 5))

    # График функции
    plt.subplot(1, 2, 1)
    plt.plot(x_values, y_values, label=r'$f(x) = 3^x - 5x^2 + 1$', color='green')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.title('График функции')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.legend()

    # График погрешностей
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(errors_iter) + 1), errors_iter, label='Простая итерация', color='blue')
    plt.scatter(range(1, len(errors_iter) + 1), errors_iter, color='blue', s=30)

    plt.plot(range(1, len(errors_Newt) + 1), errors_Newt, label='Метод Ньютона', color='red')
    plt.scatter(range(1, len(errors_Newt) + 1), errors_Newt, color='red', s=30)

    plt.yscale('log')
    plt.xlabel('Итерация')
    plt.ylabel('Погрешность')
    plt.title('Сравнение погрешностей')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
    
def main():
    eps, a, b = read_file("input.txt")

    # Проверяем условие f(a) * f(b) < 0
    if not check_condition(a, b):
        write_error_in_file("output.txt", f"Условие f(a) * f(b) < 0 не выполнено на отрезке [{a}, {b}].")
        return
    
    x_iter, iter_iter, errors_iter = iteration_method(eps, a, b)
    
    x_Newt, iter_Newt, errors_Newt = Newton_method(eps, a, b)
    
    write_in_file("output.txt", a, b, eps, x_iter, iter_iter, x_Newt, iter_Newt)
    
    plot_all(errors_iter, errors_Newt)
    

if __name__ == "__main__":
    
    main()