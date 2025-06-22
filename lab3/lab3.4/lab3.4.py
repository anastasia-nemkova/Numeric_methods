from scipy.interpolate import CubicSpline

def read_file(filename):
    with open(filename, "r") as file:
        x_test = float(file.readline())
        x = list(map(float, file.readline().split()))
        y = list(map(float, file.readline().split()))
    return x_test, x, y

def scipy_derivatives(x_test, x, y):
    cs = CubicSpline(x, y)
    df_scipy = cs(x_test, 1)
    d2f_scipy = cs(x_test, 2)
    return df_scipy, d2f_scipy

def write_results(filename, x_test, x, y, df, d2f):
    with open(filename, "w", encoding='utf-8') as file:
        df_scipy, d2f_scipy = scipy_derivatives(x_test, x, y)

        file.write(f"Первая производная\n")
        file.write(f"df({x_test}) = {df}\n")
        file.write(f"Проверка\n")
        file.write(f"df_scipy({x_test}) = {df_scipy}\n")

        file.write(f"\nВторая производная\n")
        file.write(f"d2f({x_test}) = {d2f}\n")
        file.write(f"Проверка\n")
        file.write(f"d2f_scipy({x_test}) = {d2f_scipy}\n")

def find_interval(x_test, x):
    for i in range (len(x) - 1):
        if x_test >= x[i] and x_test < x[i + 1]:
            return i

def first_derivative(x_test, x, y):
    i = find_interval(x_test, x)

    s1 = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
    s2 = ((y[i + 2] - y[i + 1]) / (x[i + 2] - x[i + 1]) - s1) / (x[i + 2] - x[i]) * (2 * x_test - x[i] - x[i + 1])
    return s1 + s2

def second_derivative(x_test, x, y):
    i = find_interval(x_test, x)

    s1 = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
    s2 = (y[i + 2] - y[i + 1]) / (x[i + 2] - x[i + 1])
    return 2 * (s2 - s1) / (x[i + 2] - x[i])
    # return (y[i] - 2 * y[i + 1] + y [i + 2]) / (x[i + 1] - x[i]) ** 2

def main():
    x_test, x, y = read_file("input.txt")

    df = first_derivative(x_test, x, y)
    d2f = second_derivative(x_test, x, y)

    write_results("output.txt", x_test, x, y, df, d2f)

if __name__ == "__main__":
    main()

