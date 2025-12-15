from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence

from comparation.types import SignalData, IndicatorResult
from comparation.indicators.rms_cv_wrapper import run_indicator_rms_cv
from comparation.indicators.STT_SVD_wrapper import run_indicator_SST_SVD
from comparation.indicators.MaxEnt_SPRT_wrapper import run_indicator_MaxEnt_SPRT
from comparation.indicators.EMD_HHT_wrapper import run_indicator_EMD_HHT

# Tipo de una función de indicador: recibe SignalData y kwargs, devuelve IndicatorResult
IndicatorFunc = Callable[..., IndicatorResult]

import matplotlib.pyplot as plt


# =========================================================
# REGISTRO CENTRAL DE INDICADORES
# =========================================================
# Aquí defines qué indicadores se van a ejecutar y con qué parámetros.
# Para añadir un nuevo método solo hay que añadir un diccionario más.

INDICATOR_CONFIG: List[Dict[str, Any]] = [
    {
        "id": "RMS_CV",                  # identificador interno (opcional)
        "func": run_indicator_rms_cv,    # wrapper del indicador
        "params": {                      # parámetros por defecto para este benchmark
            "n_max": 20,
            "samples_per_window": 3000,
            "overlap_pct": 0.0,
            "detrend": False,
            "pad_mode": "none",
            "use_unbiased_std": True,
            "eps": 1e-12,
            "cv_threshold": 0.8,
            "rms_threshold": 0.9,
            "n_min_cv": 2,
            "warmup_ignore_alerts": False,
            "start_time": 0.05,
        },
    },


    {
        "id": "SST_SVD",
        "func": run_indicator_SST_SVD,
        "params": {
            "n_fft_power": 3,
            "win_length_ms": 50.0,
            "hop_ms": 30.0,
            "Ai_length": 4,
            "mode": "causal_inclusive",
            "sigma": 6.0,
            "frac_stable": 0.36052,
            "alpha": 0.05,
            "z": 3.0,
            "fallback_mad": False,
        },
    },
    
    
    {
        "id": "MaxEnt_SPRT",
        "func": run_indicator_MaxEnt_SPRT,
        "params": {
            "rpm": 12_000.0,
            "ratio_sampling": 250.0,
            "N_seg": 10,
            "t_stable_total": 5.365770208787228,
            "alpha": 0.05,
            "beta": 0.05,
            "reset_on_H0": True,
            "cut_start_time": 0.05,
            "cut_end_time": 10,
        },
    },
    
    {
        "id": "EMD_HHT",
        "func": run_indicator_EMD_HHT,
        "params": {
            "detrend": True,
            "emd_method":"emd",          # "emd" or "eemd" also valid
            "ceemdan_noise_strength":0.2,
            "ceemdan_ensemble_size":50,
            "band_chatter":(125, 150),
            "band_selection_margin":0.15,
            "imf_selection":"index",
            "imf_index":2,
            # Hilbert
            "phase_diff_mode":"first_diff",
            "sg_window":11,
            "sg_polyorder":2,
            "f_inst_smooth_median":None,
            # HHS
            "hhs_enable":True,
            "hhs_fmax":700,
            "hhs_fbin_hz":2.0,
            "count_win_samples":500,
            "energy_mode":"A2",
            "thr_mode":"none",
            "thr_k_mad":0.0,
            "thr_percentile":50,
        },
    },
]


# =========================================================
# FUNCIONES PÚBLICAS DEL RUNNER
# =========================================================
def list_indicators() -> Sequence[str]:
    """
    Devuelve la lista de IDs de indicadores registrados en INDICATOR_CONFIG.
    """
    return [cfg["id"] for cfg in INDICATOR_CONFIG]


def run_all_indicators(signal: SignalData) -> List[IndicatorResult]:
    """
    Ejecuta TODOS los indicadores registrados en INDICATOR_CONFIG.

    Parameters
    ----------
    signal : SignalData
        Señal de entrada (t, x, fs, meta).

    Returns
    -------
    List[IndicatorResult]
        Lista de resultados, uno por indicador.
    """
    results: List[IndicatorResult] = []
    
    plt.figure(figsize=(10, 4))
    plt.plot(signal.t, signal.x, label="Tool Dyn Velocity")
    # plt.legend()
    plt.title("Tool Dynamometer Velocity Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity")
    plt.show()

    for cfg in INDICATOR_CONFIG:
        func: IndicatorFunc = cfg["func"]
        params: Dict[str, Any] = cfg.get("params", {})

        # Llamada al wrapper correspondiente
        res = func(signal, **params)

        # (Opcional) podrías verificar aquí que res.name coincide con cfg["id"]
        results.append(res)

    return results


def run_selected_indicators(signal: SignalData,
                            ids: Sequence[str]) -> List[IndicatorResult]:
    """
    Ejecuta solo un subconjunto de indicadores, especificados por sus IDs.

    Parameters
    ----------
    signal : SignalData
        Señal de entrada.
    ids : Sequence[str]
        Lista de identificadores de indicadores a ejecutar.

    Returns
    -------
    List[IndicatorResult]
        Resultados solo de los indicadores seleccionados.
    """
    results: List[IndicatorResult] = []

    id_set = set(ids)

    for cfg in INDICATOR_CONFIG:
        if cfg["id"] not in id_set:
            continue

        func: IndicatorFunc = cfg["func"]
        params: Dict[str, Any] = cfg.get("params", {})

        res = func(signal, **params)
        results.append(res)

    return results
