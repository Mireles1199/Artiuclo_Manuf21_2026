from __future__ import annotations

import signal
from typing import Optional

import numpy as np

from comparation.types import SignalData, IndicatorResult
from collections import defaultdict

from matplotlib import pyplot as plt

from typing import Sequence, Tuple, Dict, Optional, Callable, Mapping, List, Any, Union

from C_emd_hht import amplitude_spectrum
from C_emd_hht import detect_chatter_from_force, plot_tendencia
from C_emd_hht import plot_imfs, plot_imf_seleccionado, plot_tendencia, plot_HHS



def run_indicator_EMD_HHT(
    signal: SignalData,
    detrend: bool,
    emd_method: str,
    ceemdan_noise_strength: float,
    ceemdan_ensemble_size: int,
    band_chatter: Tuple[float, float],
    band_selection_margin: float,
    imf_selection: str,
    imf_index: int,
    
    phase_diff_mode: str,
    sg_window: int,
    sg_polyorder: int,
    f_inst_smooth_median: int,
    
    hhs_enable: bool,
    hhs_fmax: float,
    hhs_fbin_hz: float,
    count_win_samples: int,
    energy_mode: str,
    thr_mode: str,
    thr_k_mad: float,
    thr_percentile: float,
    

    
    ) -> IndicatorResult:
    
    
    t = signal.t
    x = signal.x
    fs = signal.fs
    fmax = fs/2.0
    
    f, A = amplitude_spectrum(x, fs=fs,)
    
    # Example/plot block (same logic as original)
    plt.figure()
    plt.plot(t, x)
    plt.title("Example force signal F(t)")
    plt.xlabel("Time (s)")
    plt.ylabel("Force (a.u.)")
    plt.show()

    plt.figure()
    plt.plot(f, A)
    plt.title("Amplitude spectrum of F(t)")
    plt.xlabel("Frequency (Hz)")
    plt.xlim(0, fmax)
    plt.ylabel("Amplitude (a.u.)")
    plt.show()
    
    res = detect_chatter_from_force(
        F=x,
        fs=fs,
        detrend=False,
        bandlimit_pre=(-100, fmax),
        emd_method=emd_method,          # "emd" or "eemd" also valid
        ceemdan_noise_strength=ceemdan_noise_strength,
        ceemdan_ensemble_size=ceemdan_ensemble_size,
        band_chatter=band_chatter,
        band_selection_margin=band_selection_margin,
        imf_selection=imf_selection,
        imf_index=imf_index,
        # Hilbert
        phase_diff_mode=phase_diff_mode,
        sg_window=sg_window,
        sg_polyorder=sg_polyorder,
        f_inst_smooth_median=f_inst_smooth_median,
        # HHS
        hhs_enable=hhs_enable,
        hhs_fmax=hhs_fmax,
        hhs_fbin_hz=hhs_fbin_hz,
        count_win_samples=count_win_samples,
        energy_mode=energy_mode,
        thr_mode=thr_mode,
        thr_k_mad=thr_k_mad,
        thr_percentile=thr_percentile,
    )
    
    # --- 3) Plot ALL IMFs in a single figure ---
    fig, axes = plot_imfs(
        imfs=res.imfs,
        plot_spectrum=True,
        fs=fs,
        f_max=hhs_fmax,
        max_to_plot=None,
        ncols=1,
        show=True
    )
    
    # --- 4) Plot selected IMF and optionally A(t) and f_inst(t) ---
    plot_imf_seleccionado(
        selected_imf=res.selected_imf,
        fs=fs,
        f_max = hhs_fmax,
        plot_spectrum=True,
        A=res.A,
        f_inst=res.f_inst,
        show=True
    )
    
    print("IMFs:", res.imfs.shape if res.imfs is not None else None)
    print("Selected IMF:", res.k_selected)
    print("Counts shape:", res.counts.shape)
    print("Threshold used (energy):", res.meta.get("threshold"))
    
    plot_HHS(
        t=t,
        fgrid=res.fgrid,
        HHS=res.HHS,
        fmax=hhs_fmax,
    )
    
    t_counts = res.t_counts_samples / fs
    counts = res.counts
    
    
    plot_tendencia(t_counts, counts, tipo="step_suave", grado=3, ls='--', lw=2, color='k')
    plt.show()

        
    
    
    
    result = IndicatorResult(
        name="EMD_HHT",  # nombre del indicador (pon el que quieras)
        t=t_counts,
        I_t=counts,
        t_d=None,
        meta={
            "Num_IMFs": res.imfs.shape[0] if res.imfs is not None else 0,
            "Selected_IMF_Index": res.k_selected,
            "IMFS": res.imfs,
            "Selected_IMF": res.selected_imf,
            "Counts_shape": res.counts.shape,
            "Threshold_used_energy": res.meta.get("threshold"),
            
            "fgrid": res.fgrid,
            "HHS": res.HHS,
            
            "fs": fs,
            "fmax": fmax,
            "emd_method": emd_method,
            "band_chatter": band_chatter,
            "imf_selection": imf_selection,
            "hhs_enable": hhs_enable,
            "energy_mode": energy_mode,
            "thr_mode": thr_mode,
            "thr_k_mad": thr_k_mad,
            "thr_percentile": thr_percentile,
            
            
            
        },
    )

    return result