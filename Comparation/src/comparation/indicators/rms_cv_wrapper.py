from __future__ import annotations

from typing import Optional

import numpy as np

from comparation.types import SignalData, IndicatorResult
from collections import defaultdict
import matplotlib.pyplot as plt

from rms_cv import plot_signal, plot_rms, plot_cv


def run_indicator_rms_cv(
    signal: SignalData,
    n_max: int,
    samples_per_window: int,
    use_unbiased_std: bool = True,
    eps: float = 1e-12,
    
    overlap_pct: Optional[float] = 0.0,
    detrend: Optional[bool] = None,
    pad_mode: Optional[str] = None,

    # ==========CV Online Config=============
    cv_threshold: Optional[float] = None,
    rms_threshold: Optional[float] = None,

    n_min_cv: int = 2,
    warmup_ignore_alerts: bool = False,

    fs_rms: Optional[float] = None,

    start_time: float = 0.0,
    
) -> IndicatorResult:
    """
    Wrapper del indicador basado en EMD (ejemplo).

    Parameters
    ----------
    signal : SignalData
        Señal de entrada (t, x, t_original, x_original, fs, meta).
    window_sec : float, Optional
        Tamaño de ventana para cálculo de RMS en segundos.
    overlap_frac : float, Optional
        Fracción de solapamiento entre ventanas (0 a 1).
    detrend : bool, Optional
        Si es True, se elimina la tendencia lineal antes de calcular RMS.
    pad_mode : str, Optional
        Modo de padding para el cálculo de RMS ('edge', 'zero', etc.).

    Returns
    -------
    IndicatorResult
        Resultado estándar del indicador, listo para ser usado en comparaciones/plots.
    """

    t = signal.t
    x = signal.x
    fs = signal.fs
    

    from rms_cv import rms_sequence, CVOnlineConfig, CVOnlineMonitor
    from rms_cv import plot_cv, plot_rms 
    
    window_sec: float = samples_per_window / fs

    dt_rms: float = window_sec * (1.0 - overlap_pct)
    
    out = rms_sequence(x, fs, window_sec=window_sec, 
                          overlap_pct=overlap_pct, detrend=detrend, 
                          pad_mode=pad_mode,
                          return_indices=True)
    rms_vals = out["rms"]
    times = out["times"]
    # plot_rms(times, rms_vals, title="RM S sequence")
    # plt.show()
    
    # ======== CV Online Monitoring ========
    cfg = CVOnlineConfig(
        n_max=n_max
        , use_unbiased_std=use_unbiased_std, eps=eps,
        cv_threshold=cv_threshold, rms_threshold=rms_threshold,
        n_min_cv=n_min_cv, warmup_ignore_alerts=warmup_ignore_alerts,
        dt_rms=dt_rms, start_time=start_time
    )
    mon = CVOnlineMonitor(cfg)
    
    results = defaultdict(list)
    for r in rms_vals:
        res = mon.update(float(r))
        for k, v in res.items():
            results[k].append(v)
            
    # plot_cv(results["time"], results["cv"], cfg.cv_threshold,  title="CV over time")
    # plt.show()
    mask = np.where(np.asarray(results["cv"]) >= cfg.cv_threshold)[0]
    chatter_points_time = np.array(results["time"])[mask]
    chatter_points_cv = np.array(results["cv"])[mask]
    

    # ======================
    # Empaquetar resultado
    # ======================
    result = IndicatorResult(
        name="RMS_CV",  # nombre del indicador (pon el que quieras)
        t=results["time"],
        I_t=results["cv"],
        # t_d=results["idx"],
        t_d=chatter_points_time,
        meta={
            "n" : results["n"],
            "mu" : results["mu"],
            "sigma" : results["sigma"],
            "alert" : results["alert"],
            "reason" : results["reason"],
            "cv_threshold": cv_threshold,
            "rms_threshold": rms_threshold,
            "n_max": n_max,
            "use_unbiased_std": use_unbiased_std,
            "eps": eps,
            "n_min_cv": n_min_cv,
            "warmup_ignore_alerts": warmup_ignore_alerts,
            "fs_rms": fs_rms,
            "dt_rms": dt_rms,
            "start_time": start_time,
            "samples_per_window": samples_per_window,
            "t_rms": times,
            "rms_values": rms_vals,
            "cv_time": results["time"],
            "cv_values": results["cv"],
            "window_sec": window_sec,
            "idx_rms_windows": out["indices"],
            
        },
    )

    return result
