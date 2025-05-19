import numpy as np
import pandas as pd

# -----------------------------------------------------------
# Hilfsfunktion: logarithmische Gewichtung g(t)  (Formel 24)
# -----------------------------------------------------------
def g(t: float, T: float, T1: float) -> float:
    """
    Logarithmisch fallende Gewichtungsfunktion g(t)
    t   : Startzeit des Vorgangs im Basisplan   (t ≥ T1)
    T   : Gesamter Planhorizont  (z. B. max(df_plan['End']))
    T1  : Rescheduling-Zeitpunkt (Ende der 1-Tages-Simulation)
    """
    denom = np.log(T) - np.log(T1)
    return (np.log(T) - np.log(t)) / denom          # identisch zu Formel 24



# -----------------------------------------------------------
#  Time-Shift-Term P_T berechnen  +  DataFrame zurückgeben
# -----------------------------------------------------------
def compute_P_T(df_plan: pd.DataFrame,
                df_revised: pd.DataFrame,
                T1: float,
                verbose: bool = True):
    """
    Liefert
    -------
    P_T     : float
        Time-Shift-Index nach Formel 22.
    details : pd.DataFrame
        Zeilenweiser Beitrag je Operation mit Spalten:
        Job, Op, Start_plan, Start_rev, delta_t, g, contrib
    """

    # 1) Zusammenführen & Filter
    details = (
        df_plan[['Job', 'Operation', 'Start']].rename(columns={'Start': 'Start_plan'})
        .merge(
            df_revised[['Job', 'Operation', 'Start']].rename(columns={'Start': 'Start_rev'}),
            on=['Job', 'Operation'], how='inner'
        )
        .query('Start_plan >= @T1')
        .assign(delta_t=lambda d: (d.Start_plan - d.Start_rev).abs())
    )

    # 2) g(t) und Beitrag
    T = max(df_plan['End'].max(), df_revised['End'].max())
    details['g']       = details['Start_plan'].apply(lambda t: g(t, T, T1))
    details['contrib'] = details['g'] * details['delta_t']
    P_T = details['contrib'].sum()

    # 3) Optionale Debug-Ausgabe (wie Version 2, leicht gekürzt)
    if verbose:
        print("=" * 70)
        print("Debug-Info  compute_P_T".center(70))
        print("=" * 70)
        print(f"{'Vorgänge nach T1':<30}: {len(details):>10}")
        print(f"{'Planungshorizont T':<30}: {T:>10.2f}")
        print("-" * 70)
        print(f"{'Metric':<25}{'Min':>12}{'Mean':>12}{'Max':>12}")
        print("-" * 70)
        print(f"{'delta_t (|t−t′|)':<25}{details['delta_t'].min():>12.2f}"
              f"{details['delta_t'].mean():>12.2f}{details['delta_t'].max():>12.2f}")
        print(f"{'g(t)':<25}{details['g'].min():>12.3f}"
              f"{details['g'].mean():>12.3f}{details['g'].max():>12.3f}")
        print("-" * 70)
        print("Beispiel-Zeilen (Top 5):")
        print(
            details[['Job', 'Operation', 'Start_plan', 'Start_rev',
                     'delta_t', 'g', 'contrib']]
            .head()
            .to_string(index=False, formatters={
                'Start_plan': '{:,.2f}'.format,
                'Start_rev' : '{:,.2f}'.format,
                'delta_t'   : '{:,.2f}'.format,
                'g'         : '{:,.3f}'.format,
                'contrib'   : '{:,.2f}'.format
            })
        )
        print("-" * 70)
        print(f"{'P_T (Summe)':<25}: {P_T:>12.2f}")
        print("=" * 70)

    return P_T, details