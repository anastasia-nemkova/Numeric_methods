def read_file(filename):
    with open(filename, "r") as file:
        x0 = float(file.readline())
        xk = float(file.readline())
        h1 = float(file.readline())
        h2 = float(file.readline())
    return x0, xk, h1, h2

def write_results(filename, h1, h2, rec_h1, rec_h2, trap_h1, trap_h2, simp_h1, simp_h2, rec_err, trap_err, simp_err):
    with open(filename, "w", encoding='utf-8') as file:
        file.write(f"Метод прямоугольника\n")
        file.write(f"Шаг {h1}: интеграл = {rec_h1}\n")
        file.write(f"Шаг {h2}: интеграл = {rec_h2}\n")
        file.write(f"Оценка погрешности: {rec_err}\n")
        
        file.write(f"\nМетод трапеции\n")
        file.write(f"Шаг {h1}: интеграл = {trap_h1}\n")
        file.write(f"Шаг {h2}: интеграл = {trap_h2}\n")
        file.write(f"Оценка погрешности: {trap_err}\n")

        file.write(f"\nМетод Симпсона\n")
        file.write(f"Шаг {h1}: интеграл = {simp_h1}\n")
        file.write(f"Шаг {h2}: интеграл = {simp_h2}\n")
        file.write(f"Оценка погрешности: {simp_err}\n")
        

def f(x):
    return x / (x ** 3 + 8)

def rectangle_method(x0, xk, h):
    integral = 0.0
    x = x0 

    while x < xk:
        integral += f(x + h / 2) * h
        x += h
        
    return integral

def trapezoid_method(x0, xk, h):
    integral = 0.0
    x = x0

    while x < xk:
        integral += (f(x) + f(x + h)) / 2 * h
        x += h

    return integral
    
def simpson_method(x0, xk, h):
    integral = 0.0
    x = x0
    
    while x < xk:
        if x + 2 * h <= xk:
            integral += (f(x) + 4 * f(x + h) + f(x + 2 * h)) * h / 3
            x += 2 * h
            
    return integral
     
def runge_romberg_method(Ih1, Ih2, h1, h2, p):
    k = h2 / h1
    return (Ih1 - Ih2) / (k ** p - 1)

def main():
    x0, xk, h1, h2 = read_file("input.txt")

    rectangle_h1 = rectangle_method(x0, xk, h1)
    rectangle_h2 = rectangle_method(x0, xk, h2)

    trapezoid_h1 = trapezoid_method(x0, xk, h1)
    trapezoid_h2 = trapezoid_method(x0, xk, h2)

    simpson_h1 = simpson_method(x0, xk, h1)
    simpson_h2 = simpson_method(x0, xk, h2)

    rec_err = abs(runge_romberg_method(rectangle_h1, rectangle_h2, h1, h2, 2))
    trap_err = abs(runge_romberg_method(trapezoid_h1, trapezoid_h2, h1, h2, 2))
    simp_err = abs(runge_romberg_method(simpson_h1, simpson_h2, h1, h2, 4))

    write_results("output.txt", h1, h2, rectangle_h1, rectangle_h2, trapezoid_h1, trapezoid_h2, simpson_h1, simpson_h2, rec_err, trap_err, simp_err)

if __name__ == "__main__":
    main()