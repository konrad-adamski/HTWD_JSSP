import pandas as pd
import numpy as np

# Generierung der Ankunftszeiten für geg. Job-Matrix ------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------

def generate_job_arrivals_df_by_mean_interarrival_time(df_jssp: pd.DataFrame,
                                                       t_a: float = 70,
                                                       random_seed_times: int = 122) -> pd.DataFrame:
    """
    Erzeugt zufällige Job-Ankunftszeiten basierend auf einer mittleren
    Interarrival-Zeit t_a und der Reihenfolge aus df_jssp.

    Parameter:
    - df_jssp: DataFrame mit mindestens der Spalte 'Job'.
    - t_a: mittlere Interarrival-Zeit für die Exponentialverteilung.
    - random_seed_times: Seed für den Zufallszahlengenerator.

    Rückgabe:
    - df_arrivals: DataFrame mit Spalten ['Job','Arrival'].
      Die Reihenfolge entspricht der Reihenfolge, in der Jobs erstmals in df_jssp erscheinen.
    """
    # 1) Liste der eindeutigen Jobs in der Reihenfolge ihres ersten Vorkommens
    job_names = df_jssp['Job'].unique().tolist()
    n_jobs = len(job_names)

    # 2) Exponentiell verteilte Interarrival-Zeiten
    np.random.seed(random_seed_times)
    interarrival = np.random.exponential(scale=t_a, size=n_jobs)

    # 3) Erste Ankunft bei 0
    interarrival[0] = 0.0

    # 4) Kumulieren und Runden
    arrival_times = np.round(np.cumsum(interarrival), 2)

    # 5) DataFrame erzeugen
    df_arrivals = pd.DataFrame({
        'Job': job_names,
        'Arrival': arrival_times
    })

    return df_arrivals


# Berechnung der mittleren Zwischenankunftszeit für geg. Job-Matrix ---------------------------------------------------------
def calculate_mean_interarrival_time(df, u_b_mmax: float = 0.9) -> float:
    """
    Berechnet die mittlere Interarrival-Zeit t_a für ein DataFrame,
    sodass die Engpassmaschine mit Auslastung u_b_mmax (< 1.0) betrieben wird.

    Parameter:
    - df: DataFrame mit Spalten ['Job','Machine','Processing Time']
    - u_b_mmax: Ziel-Auslastung der Engpassmaschine (z.B. 0.9)

    Rückgabe:
    - t_a: mittlere Interarrival-Zeit, gerundet auf 2 Dezimalstellen
    """
    # Anzahl der unterschiedlichen Jobs
    n_jobs = df['Job'].nunique()
    # Gleichverteilung über die Jobs
    p = [1.0 / n_jobs] * n_jobs

    # Vektor der Bearbeitungszeiten auf der Engpassmaschine
    vec_t_b_mmax = _get_vec_t_b_mmax(df)

    # Berechnung der mittleren Interarrival-Zeit
    t_a = sum(p[i] * vec_t_b_mmax[i] for i in range(n_jobs)) / u_b_mmax
    return round(t_a, 2)


# Vektor (der Dauer) für die Engpassmaschine
def _get_vec_t_b_mmax(df):
    """
    Ermittelt für jeden Job die Bearbeitungszeit auf der Engpassmaschine.

    Parameter:
    - df: DataFrame mit Spalten ['Job','Machine','Processing Time']

    Rückgabe:
    - Liste der Bearbeitungszeiten auf der Engpassmaschine, in der Reihenfolge
      der ersten Vorkommen der Jobs in df['Job'].
    """
    # 1) Kopie und Machine-Spalte in int umwandeln, falls nötig
    d = df.copy()
    if d['Machine'].dtype == object:
        d['Machine'] = d['Machine'].str.lstrip('M').astype(int)

    # 2) Engpassmaschine bestimmen
    eng = _get_engpassmaschine(d)

    # 3) Job-Reihenfolge festlegen
    job_order = d['Job'].unique().tolist()

    # 4) Zeiten auf Engpassmaschine extrahieren
    proc_on_eng = d[d['Machine'] == eng].set_index('Job')['Processing Time'].to_dict()

    # 5) Vektor aufbauen (0, wenn ein Job die Maschine nicht nutzt)
    vec = [proc_on_eng.get(job, 0) for job in job_order]
    return vec

# Engpassmaschine (über die gesamten Job-Matrix)
def _get_engpassmaschine(df, debug=False):
    """
    Ermittelt die Maschine mit der höchsten Gesamtbearbeitungszeit (Bottleneck) aus einem DataFrame.

    Parameter:
    - df: DataFrame mit Spalten ['Job','Machine','Processing Time']
          Machine kann entweder als int oder als String 'M{int}' vorliegen.
    - debug: Wenn True, wird die vollständige Auswertung der Maschinenbelastung ausgegeben.

    Rückgabe:
    - Index der Engpassmaschine (int)
    """
    d = df.copy()
    # Falls Machine als 'M0','M1',... vorliegt, entfernen wir das 'M'
    if d['Machine'].dtype == object:
        d['Machine'] = d['Machine'].str.lstrip('M').astype(int)
    # Gesamtbearbeitungszeit pro Maschine
    usage = d.groupby('Machine')['Processing Time'].sum().to_dict()
    if debug:
        print("Maschinenbelastung (Gesamtverarbeitungszeit):")
        for m, total in sorted(usage.items()):
            print(f"  M{m}: {total}")
    # Maschine mit maximaler Gesamtzeit
    return max(usage, key=usage.get)
