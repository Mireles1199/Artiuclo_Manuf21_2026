#%%
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple, Optional, List
from scipy.optimize import root


@dataclass
class EarlyChatterResult:
    """
    Resultado del cálculo de early chatter usando el modelo 1DOF con retardo.

    Atributos
    ---------
    t_grid : np.ndarray
        Vector de tiempos t_k usados en la aproximación frozen-time.
    a_grid : np.ndarray
        Profundidad de pasada a(t_k) en cada tiempo t_k.
    lambda_max_real : np.ndarray
        Parte real de λ_max(t_k) para cada punto de la malla.
    lambda_max : np.ndarray
        Valor complejo de λ_max(t_k) (autovalor dominante aproximado).
    t_E : Optional[float]
        Tiempo de early chatter (primer cruce Re{λ_max} = 0), o None si no hay cruce.
    a_E : Optional[float]
        Profundidad de pasada a(t_E) correspondiente, o None si no hay cruce.
    idx_cross : Optional[int]
        Índice k donde se detecta el primer cambio de signo (entre k-1 y k).
    """
    t_grid: np.ndarray
    a_grid: np.ndarray
    lambda_max_real: np.ndarray
    lambda_max: np.ndarray
    t_E: Optional[float]
    a_E: Optional[float]
    idx_cross: Optional[int]


def characteristic_equation(lambda_c: complex,
                            m: float,
                            c: float,
                            k: float,
                            Kf: float,
                            a: float,
                            T: float) -> complex:
    """
    Ecuación característica del sistema 1DOF con retardo:

        m λ^2 + c λ + k + Kf a (1 - e^{-λ T}) = 0

    Parameters
    ----------
    lambda_c : complex
        Valor complejo de λ donde se evalúa la ecuación.
    m, c, k : float
        Parámetros del modo (masa, amortiguamiento, rigidez).
    Kf : float
        Coeficiente de fuerza de corte.
    a : float
        Profundidad de pasada congelada (a_t).
    T : float
        Retardo regenerativo (periodo de paso de diente).

    Returns
    -------
    complex
        Valor de la ecuación característica en λ = lambda_c.
        La condición de raíz es characteristic_equation(...) = 0.
    """
    return (m * lambda_c**2
            + c * lambda_c
            + k
            + Kf * a * (1.0 - np.exp(-lambda_c * T)))


def solve_roots_for_a(m: float,
                      c: float,
                      k: float,
                      Kf: float,
                      a: float,
                      T: float,
                      n_guesses: int = 60,
                      re_guess: float = 0.0,
                      imag_span_factor: float = 3.0) -> List[complex]:
    """
    Resuelve numéricamente varias raíces de la ecuación característica
    para un valor dado de a, usando distintos puntos de arranque λ0.

    Estrategia:
    - Se crean n_guesses valores iniciales λ0 = re_guess + i ω,
      con ω cubriendo un rango alrededor de la frecuencia natural.
    - Para cada λ0 se aplica scipy.optimize.root en 2D (parte real, imaginaria).
    - Se filtran raíces duplicadas mediante una tolerancia.

    Parameters
    ----------
    m, c, k, Kf, a, T : float
        Parámetros físicos del sistema (ver arriba).
    n_guesses : int, optional
        Número de puntos de arranque sobre el eje imaginario.
    re_guess : float, optional
        Parte real inicial de las conjeturas de λ.
    imag_span_factor : float, optional
        Factor que controla el rango de frecuencias imaginarias
        alrededor de la frecuencia natural.

    Returns
    -------
    List[complex]
        Lista de raíces (valores de λ) encontradas (no necesariamente todas).
    """
    # Frecuencia natural aproximada (sin retardo)
    omega_n = np.sqrt(k / m)

    # Rango de frecuencias imaginarias alrededor de omega_n
    omega_min = -imag_span_factor * omega_n
    omega_max = imag_span_factor * omega_n

    imag_guesses = np.linspace(omega_min, omega_max, n_guesses)

    roots_found: List[complex] = []

    def fun_vec(z: np.ndarray) -> np.ndarray:
        """Función R^2 -> R^2 para scipy.optimize.root.

        z[0] = Re(λ), z[1] = Im(λ).
        """
        lam = z[0] + 1j * z[1]
        val = characteristic_equation(lam, m, c, k, Kf, a, T)
        return np.array([val.real, val.imag])

    for w in imag_guesses:
        z0 = np.array([re_guess, w], dtype=float)
        sol = root(fun_vec, z0, method='hybr')

        if not sol.success:
            continue

        lam = sol.x[0] + 1j * sol.x[1]

        # Filtrar raíces repetidas (tolerancia en el plano complejo)
        duplicate = False
        for r in roots_found:
            if abs(lam - r) < 1e-3:
                duplicate = True
                break

        if not duplicate:
            roots_found.append(lam)

    return roots_found


def lambda_max_for_a(m: float,
                     c: float,
                     k: float,
                     Kf: float,
                     a: float,
                     T: float,
                     **root_kwargs) -> complex:
    """
    Obtiene el autovalor dominante λ_max(a), es decir, la raíz de la
    ecuación característica con mayor parte real, para un valor dado de a.

    Parameters
    ----------
    m, c, k, Kf, a, T : float
        Parámetros físicos del sistema.
    **root_kwargs :
        Parámetros adicionales que se pasan a solve_roots_for_a
        (p.ej., n_guesses, re_guess, imag_span_factor).

    Returns
    -------
    complex
        Autovalor dominante aproximado λ_max(a).
        Si no se encuentra ninguna raíz, lanza un ValueError.
    """
    roots = solve_roots_for_a(m, c, k, Kf, a, T, **root_kwargs)

    if len(roots) == 0:
        raise ValueError("No se encontraron raíces para a = {}".format(a))

    # Elegir la raíz con mayor parte real
    roots = np.array(roots, dtype=complex)
    idx_max = np.argmax(roots.real)
    return roots[idx_max]


def ramp_a_linear(t: np.ndarray,
                  a0: float,
                  a1: float,
                  t_ramp: float) -> np.ndarray:
    """
    Rampa lineal de profundidad de pasada:

        a(t) = a0 + r t,   0 <= t <= t_ramp
        a(t) = a1,         t > t_ramp

    donde r = (a1 - a0) / t_ramp.

    Parameters
    ----------
    t : np.ndarray
        Vector de tiempos.
    a0 : float
        Profundidad inicial al inicio de la rampa.
    a1 : float
        Profundidad final al final de la rampa.
    t_ramp : float
        Duración de la rampa.

    Returns
    -------
    np.ndarray
        Valores de a(t) para cada t.
    """
    r = (a1 - a0) / t_ramp
    a = np.where(t <= t_ramp, a0 + r * t, a1)
    return a


def compute_early_chatter_time(
    m: float,
    c: float,
    k: float,
    Kf: float,
    T: float,
    a0: float,
    a1: float,
    t_ramp: float,
    dt: float,
    a_of_t: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    root_kwargs: Optional[dict] = None,
) -> EarlyChatterResult:
    """
    Calcula el tiempo de early chatter t_E usando la aproximación frozen-time.

    Procedimiento:
    -------------
    1. Define una malla de tiempos t_k = k * dt (0 <= t_k <= t_ramp).
    2. Construye a_k = a(t_k) (por defecto rampa lineal de a0 a a1).
    3. Para cada a_k, resuelve la ecuación característica y obtiene λ_max(a_k).
    4. Busca el primer cruce de Re{λ_max} de negativo a >= 0.
       Ese instante se interpola linealmente para obtener t_E.

    Parameters
    ----------
    m, c, k : float
        Parámetros del modo: masa, amortiguamiento, rigidez.
    Kf : float
        Coeficiente de fuerza de corte.
    T : float
        Retardo regenerativo (periodo de paso de diente).
    a0 : float
        Profundidad de pasada inicial de la rampa.
    a1 : float
        Profundidad de pasada final de la rampa.
    t_ramp : float
        Duración total de la rampa en segundos.
    dt : float
        Paso de tiempo para la malla frozen-time (p.ej. T o T/10).
    a_of_t : callable, optional
        Función a(t) que recibe un vector t y devuelve un vector a(t).
        Si es None, se usa una rampa lineal entre a0 y a1.
    root_kwargs : dict, optional
        Argumentos extra para lambda_max_for_a / solve_roots_for_a,
        por ejemplo: {"n_guesses": 60, "re_guess": 0.0, "imag_span_factor": 3.0}.

    Returns
    -------
    EarlyChatterResult
        Estructura con la malla de tiempos, a(t), λ_max(t) y t_E, a_E.
    """
    if root_kwargs is None:
        root_kwargs = {}

    # 1) Malla de tiempos
    n_steps = int(np.floor(t_ramp / dt)) + 1
    t_grid = np.linspace(0.0, t_ramp, n_steps)

    # 2) a(t) sobre la malla
    if a_of_t is None:
        a_grid = ramp_a_linear(t_grid, a0, a1, t_ramp)
    else:
        a_grid = a_of_t(t_grid)

    # 3) Para cada a_k, hallar λ_max
    lambda_max_vals = np.zeros_like(t_grid, dtype=complex)
    lambda_max_real = np.zeros_like(t_grid, dtype=float)

    for i, (t_k, a_k) in enumerate(zip(t_grid, a_grid)):
        lam_max = lambda_max_for_a(m, c, k, Kf, a_k, T, **root_kwargs)
        lambda_max_vals[i] = lam_max
        lambda_max_real[i] = lam_max.real
        # (Opcional) imprimir progreso:
        # print(f"t = {t_k:.4f} s, a = {a_k:.4f}, Re(lambda_max) = {lam_max.real:.4e}")

    # 4) Buscar primer cruce de negativ a >= 0
    t_E: Optional[float] = None
    a_E: Optional[float] = None
    idx_cross: Optional[int] = None

    for k_idx in range(1, len(t_grid)):
        r_prev = lambda_max_real[k_idx - 1]
        r_curr = lambda_max_real[k_idx]

        if (r_prev < 0.0) and (r_curr >= 0.0):
            # Cruce entre t_{k-1} y t_k -> interpolación lineal
            t_prev = t_grid[k_idx - 1]
            t_curr = t_grid[k_idx]

            t_E = t_prev + (t_curr - t_prev) * (-r_prev) / (r_curr - r_prev)
            # Interpolar también a(t)
            a_prev = a_grid[k_idx - 1]
            a_curr = a_grid[k_idx]
            a_E = a_prev + (a_curr - a_prev) * (t_E - t_prev) / (t_curr - t_prev)

            idx_cross = k_idx
            break

    return EarlyChatterResult(
        t_grid=t_grid,
        a_grid=a_grid,
        lambda_max_real=lambda_max_real,
        lambda_max=lambda_max_vals,
        t_E=t_E,
        a_E=a_E,
        idx_cross=idx_cross,
    )

import numpy as np

import numpy as np
from typing import Optional, Tuple

def compute_visible_chatter_time(
    result: EarlyChatterResult,
    R: float = 10.0,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Calcula el tiempo t_vis en el que la amplitud ha crecido un
    factor R respecto a t_E, usando la aproximación lineal:

        A(t) ~ A(t_E) * exp( ∫_{t_E}^{t} sigma(τ) dτ ),
        sigma(t) = Re{lambda_max(t)}.

    Busca t_vis tal que:
        ∫_{t_E}^{t_vis} sigma(τ) dτ = ln(R).

    Además devuelve la ganancia equivalente en dB:

        gain_db = 20 * log10(R).

    Parameters
    ----------
    result : EarlyChatterResult
        Resultado devuelto por compute_early_chatter_time.
        Debe tener t_E y idx_cross definidos.
    R : float
        Factor de crecimiento de amplitud deseado (R > 1).
        Ejemplos típicos: 2, 5, 10, 20.

    Returns
    -------
    t_vis : float or None
        Tiempo donde se alcanza el factor R, o None si nunca se alcanza.
    a_vis : float or None
        Profundidad de pasada a(t_vis), o None si nunca se alcanza.
    gain_db : float or None
        Ganancia en dB correspondiente a R (20*log10(R)),
        o None si no se alcanza el factor R (coherente con t_vis=None).
    """
    # Si no hay cruce de estabilidad, no habrá crecimiento exponencial
    if result.t_E is None or result.idx_cross is None:
        return None, None, None

    if R <= 1.0:
        raise ValueError("R debe ser > 1 (factor de crecimiento).")

    t = result.t_grid
    sigma = result.lambda_max_real
    a = result.a_grid

    # Índice donde se detectó el cruce (entre idx_cross-1 y idx_cross)
    k0 = result.idx_cross

    # Valores en los puntos antes y después del cruce
    t_prev = t[k0 - 1]
    t_curr = t[k0]
    sigma_prev = sigma[k0 - 1]
    sigma_curr = sigma[k0]

    # sigma(t_E) aproximado por interpolación lineal
    if t_curr != t_prev:
        sigma_E = sigma_prev + (sigma_curr - sigma_prev) * \
            (result.t_E - t_prev) / (t_curr - t_prev)
    else:
        sigma_E = sigma_curr

    # Interpolamos también a(t) en t_E
    if t_curr != t_prev:
        a_E = a[k0 - 1] + (a[k0] - a[k0 - 1]) * \
            (result.t_E - t_prev) / (t_curr - t_prev)
    else:
        a_E = a[k0]

    # Construimos una malla desde t_E hacia adelante
    t_seg = np.concatenate(([result.t_E], t[k0:]))
    sigma_seg = np.concatenate(([sigma_E], sigma[k0:]))
    a_seg = np.concatenate(([a_E], a[k0:]))

    # Integral acumulada G(t) ≈ ∫_{t_E}^{t} sigma(τ) dτ (regla del trapecio)
    dt = np.diff(t_seg)
    sigma_mid = 0.5 * (sigma_seg[:-1] + sigma_seg[1:])
    G = np.concatenate(([0.0], np.cumsum(sigma_mid * dt)))

    target = np.log(R)

    # Buscamos el primer índice donde G >= ln(R)
    idx = np.where(G >= target)[0]
    if len(idx) == 0:
        # Nunca se alcanza ese nivel de crecimiento
        return None, None, None

    k = idx[0]
    if k == 0:
        # Justo en t_E ya supera (muy raro, pero por completitud)
        gain_db = 20.0 * np.log10(R)
        return t_seg[0], a_seg[0], gain_db

    # Interpolación lineal entre k-1 y k en G(t)
    G_prev, G_curr = G[k - 1], G[k]
    t_p, t_c = t_seg[k - 1], t_seg[k]
    a_p, a_c = a_seg[k - 1], a_seg[k]

    if G_curr == G_prev:
        t_vis = t_c
    else:
        t_vis = t_p + (t_c - t_p) * (target - G_prev) / (G_curr - G_prev)

    # Interpolamos también a(t) en t_vis
    if t_c == t_p:
        a_vis = a_c
    else:
        a_vis = a_p + (a_c - a_p) * (t_vis - t_p) / (t_c - t_p)

    # Ganancia en dB asociada al factor R
    gain_db = 20.0 * np.log10(R)

    return t_vis, a_vis, gain_db

#%%
# -------------------------------------------------------------------
# EJEMPLO DE USO
# -------------------------------------------------------------------
# if __name__ == "__main__":
f2   = 150.0
xsi2 = 0.01
k2   = 2.13e8
theta_2 = 135.0 * np.pi / 180.0

omega2 = 2.0 * np.pi * f2
m2 = k2 / omega2**2           # ya lo haces tú

c2 = 2.0 * xsi2 * m2 * omega2
phi2_z = phi2_z = np.sin(theta_2)

m = m2
c = c2
k = k2

Kf_fisico = 1.0e9       # N/m^2 (ejemplo genérico)
Kf_modal = (phi2_z**2) * Kf_fisico
Nz = 1           # número de dientes
n_rpm = 12093.99536  # rpm
fr = n_rpm / 60.0
T = 1.0 / (fr * Nz)

# Rampa de profundidad: de 0.1 mm a 3.0 mm en 1 segundo
a0 = 5.0e-3
a1 = 15.0e-3
f_tooth = 0.05 # mm/tooth
vf = n_rpm * f_tooth/1e3/60  # m/s
L_cylindre = 150.e-3
t_ramp = L_cylindre / vf

print("--------------------------------------------------")
print("Parámetros del modelo 1DOF con retardo:")
print(f"m = {m} kg")
print(f"c = {c} N·s/m")
print(f"k = {k} N/m")
print(f"Kf_modal = {Kf_modal} N/m^2")
print(f"T = {T} s")
print(f"Rampa de a: {a0*1e3} mm a {a1*1e3} mm en {t_ramp} s")


# Paso de tiempo para frozen-time (por ejemplo, T/2)
dt = T / 5.0

# Parámetros para la búsqueda de raíces
root_options = {
    "n_guesses": 80,
    "re_guess": -50.0,       # arrancar con parte real algo negativa
    "imag_span_factor": 4.0  # barrer ±4 * ω_n
}

result = compute_early_chatter_time(
    m=m,
    c=c,
    k=k,
    Kf=Kf_modal,
    T=T,
    a0=a0,
    a1=a1,
    t_ramp=t_ramp,
    dt=dt,
    a_of_t=None,        # usa rampa lineal
    root_kwargs=root_options,
)

#%%
percent_t_stable = result.t_E / t_ramp * 1 if result.t_E is not None else None

print("--------------------------------------------------")
print("Resultados early chatter (modelo 1DOF frozen-time)")
print("--------------------------------------------------")
print(f"t_E  = {result.t_E}  [s]")
print(f"a_E  = {result.a_E}  [m]")
print(f"% t_ramp estable = {percent_t_stable} ")
print(f"idx_cross = {result.idx_cross}")
print("--------------------------------------------------")

#%%
# -------------- Chatter visible (factor R) ----------------ù
for R in [2.0, 2.5, 3.0, 5.0, 10.0, 20.0]:
    t_vis, a_vis, gain_db = compute_visible_chatter_time(result, R=R)
    print(f"R = {R:4.1f} -> t_vis = {t_vis} s, a_vis = {a_vis} m, gain_db = {gain_db} dB")