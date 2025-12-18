import numpy as np
import cmath
from scipy import integrate
from scipy.optimize import differential_evolution, NonlinearConstraint
from datetime import datetime

# Константы
ETA = 120 * np.pi
TAU = np.pi / 2 + 1.3j
C = 3e8
ETA0 = 377


def compute_powers(params, I0=1.0, f=10e9):
    X, A, B = params

    if abs(B) < 1e-12:
        return 0.0, np.inf

    lambda_val = C / f
    k = 2 * np.pi / lambda_val
    r = 10.0 * lambda_val
    Omega = k * r
    C_const = I0 * k / (2 * np.pi * ETA0)

    # ПОВЕРХНОСТНАЯ ВОЛНА
    def surface_integrand(z):
        denominator = (-2j * X * A * cmath.cos(TAU) -
                       3 * B * ETA0 * cmath.cos(TAU) ** 2 +
                       ETA0 * (1 + B))

        exp_factor = cmath.exp(-1j * k * z * cmath.cos(TAU))

        E_z_sw = (1j * I0 * k * ETA0 *
                  (cmath.cos(TAU) * (1 + B * cmath.sin(TAU) ** 2) / denominator) *
                  cmath.exp(-1j * Omega * cmath.sin(TAU))) * exp_factor

        H_y_sw = (-2 * np.pi * 1j * C_const *
                  (ETA0 * cmath.cos(TAU) * (1 + B * cmath.sin(TAU) ** 2) /
                   (cmath.sin(TAU) * denominator)) *
                  cmath.exp(-1j * Omega * cmath.sin(TAU))) * exp_factor

        S_x = 0.5 * np.real(E_z_sw * np.conj(H_y_sw))
        return S_x

    # ОБЪЕМНАЯ ВОЛНА
    def surface_impedance(phi):
        return 1j * X * ((1 + A * cmath.sin(phi) ** 2) / (1 + B * cmath.sin(phi) ** 2))

    def volume_integrand(phi):
        Z_s = surface_impedance(phi)

        A_amp = - (C_const * np.sqrt(2 * np.pi) / cmath.exp(-1j * np.pi / 4)) * \
                (cmath.cos(phi) / ((Z_s / ETA0) + cmath.cos(phi)))

        A_amp_e = (C_const * np.sqrt(2 * np.pi) / cmath.exp(-1j * np.pi / 4)) * \
                  ((cmath.cos(phi) * Z_s) / ((Z_s / ETA0) + cmath.cos(phi)))

        E_z = (-ETA0 / np.sqrt(Omega)) * A_amp * np.sin(phi) * cmath.exp(-1j * Omega)
        E_x = (A_amp_e / np.sqrt(Omega)) * cmath.exp(-1j * Omega)
        H_y = (A_amp / np.sqrt(Omega)) * cmath.exp(-1j * Omega)

        S_x = -0.5 * (E_z * np.conj(H_y))
        S_z = 0.5 * (E_x * np.conj(H_y))
        S_normal = S_x * np.sin(phi) + S_z * np.cos(phi)
        return np.real(S_normal)

    try:
        P_surface, _ = integrate.quad(surface_integrand, 0, 1000 * lambda_val,
                                      limit=1000, epsabs=1e-12, epsrel=1e-12)

        P_volume, _ = integrate.quad(volume_integrand, -np.pi / 2, np.pi / 2,
                                     limit=1000, epsabs=1e-12, epsrel=1e-12)

        P_surface = abs(P_surface)
        P_volume = abs(P_volume)

        return float(P_surface), float(P_volume)
    except:
        return 0.0, np.inf


def constraint_equation(params):
    X, A, B = params

    if abs(B) < 1e-12:
        return np.inf

    equation = (np.cos(TAU) ** 3 +
                ((1j * X * A) / (ETA * B)) * np.cos(TAU) ** 2 +
                ((1 - B) / B) * np.cos(TAU) +
                (1j * X * (1 - A)) / (ETA * B))

    return float(abs(equation))


def two_stage_optimization():
    print("ДВУХЭТАПНАЯ ОПТИМИЗАЦИЯ")
    print("=" * 60)

    # Этап 1: Дисперсионное уравнение
    print("\nЭтап 1: Удовлетворение дисперсионного уравнения...")

    bounds = [
        (-5 * ETA, 5 * ETA),
        (-5.0, 5.0),
        (-5.0, 5.0)
    ]

    def stage1_objective(params):
        return constraint_equation(params)

    result_stage1 = differential_evolution(
        stage1_objective,
        bounds=bounds,
        maxiter=5000,
        popsize=30,
        tol=1e-12,
        disp=False,
        seed=42,
        atol=1e-10,
        workers=1
    )

    if result_stage1.fun > 1e-6:
        print(f"Этап 1 провален. Ошибка: {result_stage1.fun:.2e}")
        return None

    initial_guess = result_stage1.x
    print(f"Этап 1 завершен. Ошибка: {result_stage1.fun:.2e}")
    print(f"Параметры: X={initial_guess[0]:.2f}, A={initial_guess[1]:.4f}, B={initial_guess[2]:.4f}")

    # Этап 2: Оптимизация мощностей
    print("\nЭтап 2: Максимизация P_surface/P_volume...")

    # БЕЗОПАСНЫЕ ГРАНИЦЫ - проверка на корректность
    X_min = min(initial_guess[0] * 0.5, initial_guess[0] * 1.5)
    X_max = max(initial_guess[0] * 0.5, initial_guess[0] * 1.5)
    A_min = initial_guess[1] - 1.0
    A_max = initial_guess[1] + 1.0
    B_min = initial_guess[2] - 1.0
    B_max = initial_guess[2] + 1.0

    # Корректировка границ, если они выходят за пределы исходных
    X_min = max(X_min, -5 * ETA)
    X_max = min(X_max, 5 * ETA)
    A_min = max(A_min, -5.0)
    A_max = min(A_max, 5.0)
    B_min = max(B_min, -5.0)
    B_max = min(B_max, 5.0)

    narrow_bounds = [
        (X_min, X_max),
        (A_min, A_max),
        (B_min, B_max)
    ]

    print(f"\nГраницы второго этапа:")
    print(f"  X: [{X_min:.2f}, {X_max:.2f}]")
    print(f"  A: [{A_min:.4f}, {A_max:.4f}]")
    print(f"  B: [{B_min:.4f}, {B_max:.4f}]")

    disp_constraint = NonlinearConstraint(
        constraint_equation,
        lb=0,
        ub=1e-6
    )

    def stage2_objective(params):
        P_surface, P_volume = compute_powers(params)

        if P_surface <= 0 or P_volume <= 0:
            return 1e6

        if np.isinf(P_surface) or np.isinf(P_volume):
            return 1e6

        ratio = P_surface / P_volume

        if P_surface < 1e-12:
            surface_penalty = 100.0
        elif P_surface < 1e-9:
            surface_penalty = 10.0 * (1e-9 - P_surface) / 1e-9
        else:
            surface_penalty = 0.0

        if ratio < 1.0:
            ratio_penalty = 5.0 * (1.0 - ratio)
        else:
            ratio_penalty = 0.0

        objective_value = np.log10(ratio) - surface_penalty - ratio_penalty

        if np.isnan(objective_value):
            return 1e6

        return -objective_value

    result_stage2 = differential_evolution(
        stage2_objective,
        bounds=narrow_bounds,
        constraints=disp_constraint,
        maxiter=5000,
        popsize=30,
        tol=1e-10,
        disp=True,
        seed=42,
        mutation=(0.3, 1.0),
        recombination=0.8,
        updating='immediate',
        workers=1,
        polish=True
    )

    if result_stage2.success or result_stage2.nit > 100:
        optimal_params = result_stage2.x
        eq_error = constraint_equation(optimal_params)
        P_surface, P_volume = compute_powers(optimal_params)

        print("\n" + "=" * 60)
        print("РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ")
        print("=" * 60)
        print(f"Параметры:")
        print(f"  X = {optimal_params[0]:.8e}")
        print(f"  A = {optimal_params[1]:.8f}")
        print(f"  B = {optimal_params[2]:.8f}")
        print(f"\nМощности:")
        print(f"  P_surface = {P_surface:.8e} Вт")
        print(f"  P_volume  = {P_volume:.8e} Вт")

        if P_volume > 0:
            ratio = P_surface / P_volume
            print(f"  Отношение = {ratio:.8e}")
            print(f"  Доля SW = {P_surface / (P_surface + P_volume) * 100:.6f}%")
            print(f"  Подавление = {20 * np.log10(ratio):.2f} дБ")

        print(f"  Ошибка уравнения = {eq_error:.8e}")

        return optimal_params, P_surface, P_volume

    print("Этап 2 не сошелся.")
    return None


def brute_force_test():
    """Прямой перебор для проверки"""
    print("\n" + "=" * 60)
    print("ПРЯМОЙ ПЕРЕБОР ПАРАМЕТРОВ")
    print("=" * 60)

    best_ratio = 0
    best_params = None

    # Тестовые значения
    X_test = [100, 500, 1000, 1500, 2000]
    A_test = [-2, -1, 0, 1, 2]
    B_test = [-2, -1, 0.5, 1, 2]

    for X in X_test:
        for A in A_test:
            for B in B_test:
                if abs(B) < 0.1:
                    continue

                eq_err = constraint_equation([X, A, B])
                if eq_err > 1e-3:
                    continue

                P_surface, P_volume = compute_powers([X, A, B])

                if P_surface > 0 and P_volume > 0:
                    ratio = P_surface / P_volume
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_params = [X, A, B]
                        print(f"Найдено: X={X}, A={A}, B={B}, ratio={ratio:.2e}")

    if best_params:
        print(f"\nЛучшее отношение: {best_ratio:.2e}")
        return best_params
    return None


if __name__ == "__main__":
    print("ЗАПУСК ОПТИМИЗАЦИИ")
    print("=" * 60)

    # Сначала прямой перебор для понимания
    test_params = brute_force_test()

    if test_params:
        print(f"\nЗапуск оптимизации с начальной точки: {test_params}")
        # Можно использовать test_params как начальное приближение
    else:
        print("\nПрямой перебор не дал результатов. Запуск полной оптимизации...")

    # Запуск двухэтапной оптимизации
    results = two_stage_optimization()

    if results:
        optimal_params, P_surface, P_volume = results

        # Сохранение
        with open('results.txt', 'w') as f:
            f.write(f"X = {optimal_params[0]:.12e}\n")
            f.write(f"A = {optimal_params[1]:.12f}\n")
            f.write(f"B = {optimal_params[2]:.12f}\n")
            f.write(f"P_surface = {P_surface:.12e}\n")
            f.write(f"P_volume = {P_volume:.12e}\n")
            if P_volume > 0:
                f.write(f"Ratio = {P_surface / P_volume:.12e}\n")

        print("\nРезультаты сохранены в 'results.txt'")
    else:
        print("\nОптимизация не удалась.")