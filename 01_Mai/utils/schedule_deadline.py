import pandas as pd


def find_k(jobs: dict, arrivals: pd.DataFrame, schedule_func, target_service=0.95, buffer_factor: float = 1.0):
    """sucht k via Binärsuche; Schedule einmal vorterminiert"""
    
    # 1) einmaligen Schedule holen (für z.B. FCFS o.ä., der nicht deadline‐abhängig ist)
    sched = schedule_func(jobs, arrivals)
    
    # 2) Binärsuche
    lo, hi = 0.5, 10.0                      # Startintervall (evtl. anpassen)
    for _ in range(30):                     # 30 Iterationen ≈ 1/2^30 Genauigkeit            
        k = (lo + hi) / 2
        d = _calc_due_dates(jobs, arrivals, k)
        
        # Service‐Level neu berechnen
        on_time = (sched["End"] <= sched["Job"].map(d)).mean()
        if on_time >= target_service:
            hi = k                          # Deadlines sind noch zu großzügig
        else:
            lo = k                          # Deadlines zu eng

    # Deadlines * Faktor
    if buffer_factor > 1:
        d = _calc_due_dates(jobs, arrivals, k, buffer_factor=buffer_factor)
    return k, d


def _calc_due_dates(jobs: dict, arrivals: pd.DataFrame, k: float, buffer_factor: float = 1.0) -> dict:
    """
    Berechnet Deadlines für jedes Job j als
      d_j = a_j + (k + buffer)*p_j
    """
    # Gesamtbearbeitungszeiten p_j
    p_tot = {j: sum(d for _, d in ops) for j, ops in jobs.items()}
    # Ankunftszeiten a_j
    a = arrivals.set_index("Job")["Arrival"].to_dict()
    # Deadline-Berechnung
    return {j: a[j] + (k * p_tot[j]) * buffer_factor for j in jobs}