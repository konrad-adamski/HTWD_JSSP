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




def compute_P_T(df_plan: pd.DataFrame,
                df_revised: pd.DataFrame,
                T1: float,
                horizon_mode: str = "fixed",
                verbose: bool = True):
    """
    P_T   – Time-Shift-Index (Formel 22)

    horizon_mode
        "fixed" : T = T1 + window   (Fenstergröße konstant)
        "union" : T = max(End_plan, End_rev)
    """

    # --------------------------------------------------
    # 1) Merge & Filter  (nur Jobs + Machine-Kombi, die in beiden Plänen vorkommt)
    # --------------------------------------------------
    details = (
        df_plan[['Job', 'Machine', 'Start']].rename(columns={'Start': 'Start_plan'})
        .merge(
            df_revised[['Job', 'Machine', 'Start']].rename(columns={'Start': 'Start_rev'}),
            on=['Job', 'Machine'], how='inner'
        )
        .query('Start_plan >= @T1')                          # „frozen zone“ ausschließen
        .assign(delta_t=lambda d: (d.Start_plan - d.Start_rev).abs())
    )

    # --------------------------------------------------
    # 2) Planungshorizont T ermitteln
    # --------------------------------------------------
    if horizon_mode == 'fixed':
        window = df_plan['End'].max() - df_plan['Start'].min()
        T      = T1 + window
    else:                     # 'union'
        T      = max(df_plan['End'].max(), df_revised['End'].max())

    # --------------------------------------------------
    # 3) Gewichtung und Beitrag
    # --------------------------------------------------
    details['g']       = details['Start_plan'].apply(lambda t: g(t, T, T1))
    details['contrib'] = details['g'] * details['delta_t']
    P_T = details['contrib'].sum()

    # --------------------------------------------------
    # 4) Optionale Debug-Ausgabe
    # --------------------------------------------------
    if verbose:
        print("="*70)
        print("Debug-Info  compute_P_T".center(70))
        print("="*70)
        print(f"{'Vorgänge nach T1':<30}: {len(details):>10}")
        print(f"{'Planungshorizont T':<30}: {T:>10.2f}")
        print("-"*70)
        print(f"{'Metric':<25}{'Min':>12}{'Mean':>12}{'Max':>12}")
        print("-"*70)
        print(f"{'delta_t (|t−t′|)':<25}{details['delta_t'].min():>12.2f}"
              f"{details['delta_t'].mean():>12.2f}{details['delta_t'].max():>12.2f}")
        print(f"{'g(t)':<25}{details['g'].min():>12.3f}"
              f"{details['g'].mean():>12.3f}{details['g'].max():>12.3f}")
        print("-"*70)
        print("Beispiel-Zeilen (Top 5):")
        print(
            details[['Job','Machine','Start_plan','Start_rev',
                     'delta_t','g','contrib']]
            .head()
            .to_string(index=False, formatters={
                'Start_plan': '{:,.2f}'.format,
                'Start_rev' : '{:,.2f}'.format,
                'delta_t'   : '{:,.2f}'.format,
                'g'         : '{:,.3f}'.format,
                'contrib'   : '{:,.2f}'.format
            })
        )
        print("-"*70)
        print(f"{'P_T (Summe)':<25}: {P_T:>12.2f}")
        print("="*70)

    return P_T, details

