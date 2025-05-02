import pandas as pd
import numpy as np

# Generierung der Ankunftszeiten für geg. Job-Matrix ------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------

def generate_job_arrivals_df_by_mean_interarrival_time(job_dict, t_a= 70, random_seed_times=122):
    job_names = list(job_dict.keys())
    n_jobs = len(job_names)

    # Erzeuge Ankunftszeiten
    np.random.seed(random_seed_times)
    interarrival_times = np.random.exponential(scale=t_a, size=n_jobs)
    interarrival_times[0] = 0.0  # Start bei 0 Minuten
    arrival_times = np.round(np.cumsum(interarrival_times), 2)

    df_arrivals = pd.DataFrame({
        "Job-ID": job_names,
        "Ankunftszeit (Minuten)": arrival_times
    })

    return df_arrivals

def generate_job_arrivals_df(job_dict, u_b_mmax=0.9, random_seed_jobs=12, random_seed_times=122):
    job_names = list(job_dict.keys())
    n_jobs = len(job_names)

    # Permutiere Jobnamen
    np.random.seed(random_seed_jobs)
    shuffled_jobs = list(np.random.permutation(job_names))

    # Interarrival-Zeit auf Basis der Engpassmaschine
    t_a = calculate_mean_interarrival_time(job_dict, u_b_mmax=u_b_mmax)

    # Erzeuge Ankunftszeiten
    np.random.seed(random_seed_times)
    interarrival_times = np.random.exponential(scale=t_a, size=n_jobs)
    interarrival_times[0] = 0.0  # Start bei 0 Minuten
    arrival_times = np.round(np.cumsum(interarrival_times), 2)

    df_day = pd.DataFrame({
        "Job-ID": shuffled_jobs,
        "Ankunftszeit (Minuten)": arrival_times
    })

    return df_day
    
# Berechnung der mittleren Zwischenankunftszeit für geg. Job-Matrix -----------------------------------------
def calculate_mean_interarrival_time(jobs: dict, u_b_mmax: float = 0.9) ->float:
    """
    Berechnet die mittlere Interarrival-Zeit t_a, sodass die Engpassmaschine
    mit Auslastung u_b_mmax (< 1.0) betrieben wird.
    """
    n_jobs = len(jobs)
    p = [1 / n_jobs] * n_jobs
    vec_t_b_mmax = _get_vec_t_b_mmax(jobs)
    t_a = sum(p[i] * vec_t_b_mmax[i] for i in range(n_jobs)) / u_b_mmax
    return np.round(t_a, 2)
    
# Vektor (der Dauer) für die Engpassmaschine
def _get_vec_t_b_mmax(jobs: dict) -> list:
    # Engpassmaschine bestimmen
    engpassmaschine = _get_engpassmaschine(jobs)

    # Vektor der Bearbeitungszeiten auf der Engpassmaschine
    vec_t_b_mmax = []
    for job in jobs.values():
        duration = next((d for m, d in job if m == engpassmaschine), 0)
        vec_t_b_mmax.append(duration)

    return vec_t_b_mmax

# Engpassmaschine (über die gesamten Job-Matrix)
def _get_engpassmaschine(jobs: dict, debug=False):
    machine_usage = {}
    for job_ops in jobs.values():
        for machine, duration in job_ops:
            machine_usage[machine] = machine_usage.get(machine, 0) + duration
    if debug:
        print("\nEndstand Maschinenbelastung:", machine_usage)
    return max(machine_usage, key=machine_usage.get)