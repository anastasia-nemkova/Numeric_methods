import math

# Функция y = sin(x) + x
def f(x):
    return math.sin(x) + x

def read_file(filename):
    with open(filename, "r") as file:
        x_true = float(file.readline())
    return x_true

def lagrange_interpolation(x, y, x_true):
    '''
    ln_j = П(i != j) (x - xj) / (xi - xj)
    Pn = Сумма (yj*lnj)
    '''

    n = len(x)
    result = 0.0
    for i in range(n):
        term = y[i]
        for j in range(n):
            if i != j:
                term *= (x_true - x[j]) / (x[i] - x[j])
        result += term

    return result

def lagrange_interpolation_str(x, y):
    n = len(x)
    terms = []
    for i in range(n):
        numerator = []
        denominator = 1
        for j in range(n):
            if i != j:
                numerator.append(f"(x - {x[j]:.4f})")
                denominator *= (x[i] - x[j])

        # coeff = y[i] / denominator if denominator != 0 else float('inf')
        # term = f"{coeff:.4f}"

        term = f"{y[i]:.4f}"
        if numerator:
            term += " * " + " * ".join(numerator)
        if denominator != 1:
            term += f" / {denominator:.4f}"
        terms.append(term)
    return " + ".join(terms)

def divided_diff(x, y):
    n = len(x)
    coeff = [y[i] for i in range(n)]
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            coeff[i] = (coeff[i] - coeff[i - 1]) / (x[i] - x[i - j])
    return coeff 

def newton_interpolation(x, y, x_true):
    n = len(x)

    coeff = divided_diff(x, y)
    result = 0.0

    term = 1.0
    
    for i in range(n):
        result += term * coeff[i]
        term *= (x_true - x[i])
        
    return result

def newton_interpolation_str(x, y):
    coeff = divided_diff(x, y)

    terms = [f"{coeff[0]:.3f}"]
    
    for i in range(1, len(coeff)):
        term = f"{coeff[i]:.3f}"
        for j in range(i):
            term += f" * (x - {x[j]:.3f})"
        terms.append(term)
    return " + ".join(terms)

def solve(xi, x_true, label, file):
    file.write(f"{label}")

    yi = [f(x) for x in xi]
    y_true = f(x_true)

    y_lagr = lagrange_interpolation(xi, yi, x_true)
    y_newt = newton_interpolation(xi, yi, x_true)

    error_lagr = abs(y_lagr - y_true)
    error_newt = abs(y_newt - y_true)

    file.write("\n--- Многочлен Лагранжа ---\n")
    file.write(lagrange_interpolation_str(xi, yi) + "\n")

    file.write("\n--- Многочлен Ньютона ---\n")
    file.write(newton_interpolation_str(xi, yi) + "\n")

    file.write(f"\nЗначение функции в x* = {x_true} : {y_true:.8f}\n")
    file.write(f"Интерполяция Лагранжа: {y_lagr:.8f}, Погрешность: {error_lagr:.4e}\n")
    file.write(f"Интерполяция Ньютона : {y_newt:.8f}, Погрешность: {error_newt:.4e}\n")


def write_in_file(filename, xi_a, xi_b, x_true):
    with open("output.txt", "w", encoding="utf-8") as file:
        file.write("Функция: y = sin(x) + x\n")
        file.write(f"\nТочка интерполяции x* = {x_true}\n")
        solve(xi_a, x_true, "а) Xi = 0, π/6, 2π/6, 3π/6", file)
        solve(xi_b, x_true, "б) Xi = 0, π/6, π/4, π/2", file)


def main():
    x_true = read_file("input.txt")

    xi_a = [0, math.pi/6, math.pi * 2 / 6, math.pi * 3 / 6]
    xi_b = [0, math.pi / 6, math.pi / 4, math.pi / 2]

    write_in_file("output.txt", xi_a, xi_b, x_true)



if __name__ == "__main__":
    main()


