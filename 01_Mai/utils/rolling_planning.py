import re
import random
import numpy as np
import pandas as pd

import utils.schedule_solver__with_arrivals as ssv
import utils.schedule_deadline as sdead



import utils.schedule_interarrival as sit

def init_jobs_with_arrivals(new_job_dict: dict, day_num: int, u_b_mmax: float = 0.9, 
                            generate_deadlines: bool = False, deadlines_buffer_factor: float = 2.0) -> (dict, pd.DataFrame):

    # Startwerte: keine alten Jobs oder Ankünfte
    old_job_dict = {}
    old_arrivals_df = pd.DataFrame(columns=["Job", "Arrival"])

    combined_jobs = {}
    arrivals_list = []

    # Für jeden Tag neu erzeugen und anhängen
    for _ in range(day_num):
        new_jobs, df_arrivals = create_new_jobs_with_arrivals_for_one_day(old_job_dict, old_arrivals_df, 
                                                                          new_job_dict, u_b_mmax=u_b_mmax, generate_deadlines = generate_deadlines,
                                                                          deadlines_buffer_factor = deadlines_buffer_factor)
        # Jobs sammeln
        combined_jobs.update(new_jobs)
        # Ankünfte sammeln
        arrivals_list.append(df_arrivals)

        # Für den nächsten Tag als Basis übernehmen
        old_job_dict = new_jobs
        old_arrivals_df = df_arrivals

    # Alle Ankünfte zu einem DataFrame zusammenführen
    df_all_arrivals = pd.concat(arrivals_list, ignore_index=True)

    return combined_jobs, df_all_arrivals



def update_new_day(existing_jobs: dict, existing_arrivals_df: pd.DataFrame, new_job_dict: dict, u_b_mmax: float = 0.9, 
                   shuffle: bool = False, generate_deadlines: bool = False, deadlines_buffer_factor: float = 2.0):

    # 1) Neue Jobs + Ankünfte für einen Tag erzeugen
    new_jobs, df_arrivals_new = create_new_jobs_with_arrivals_for_one_day(existing_jobs, existing_arrivals_df, new_job_dict, 
                                                                          u_b_mmax=u_b_mmax, shuffle=shuffle, generate_deadlines = generate_deadlines, 
                                                                          deadlines_buffer_factor = deadlines_buffer_factor)
    
    # 2) Bestehende Jobs um die Neuen erweitern
    existing_jobs.update(new_jobs)

    # 3) Bestehende Ankünfte an die Neuen anhängen
    updated_arrivals_df = pd.concat([existing_arrivals_df, df_arrivals_new], ignore_index=True)

    return existing_jobs, updated_arrivals_df

# -------------------------------------------------------------------------------------------------------------------------
def create_new_jobs_with_arrivals_for_one_day(old_job_dict: dict, old_arrivals_df: pd.DataFrame, new_job_dict: dict, 
                                              u_b_mmax: float = 0.9, shuffle: bool = False, generate_deadlines: bool = False, 
                                              deadlines_buffer_factor: float = 2.0):

    new_day_start = 0
    
    if old_arrivals_df is not None and not old_arrivals_df.empty:
        last_old_arrival = old_arrivals_df["Arrival"].max()
        new_day_start = ((last_old_arrival // 1440) + 1) * 1440

        
    # 1) Instanz vervielfachen
    new_jssp_data = {}
    job_dict_temp = old_job_dict.copy() if old_job_dict else {}

    
    for i in range(3):
        
        # abwechselnd shufflen oder nicht
        shuffle_flag = shuffle if (i % 2 == 0) else not shuffle
        
        # erzeugt ein dict neuer Jobs
        new_jobs_i = create_new_jobs(job_dict_temp, new_job_dict, shuffle=shuffle_flag, seed=50)
        job_dict_temp  = new_jobs_i
        
        new_jssp_data.update(new_jobs_i)

    # 2) mittlere Zwischenankunftszeit berechnen
    mean_interarrival_time = sit.calculate_mean_interarrival_time(new_jssp_data, u_b_mmax)

    # 3) Ankunftszeiten generieren
    new_arrival_times = create_new_arrivals(len(new_jssp_data), mean_interarrival_time, new_day_start, random_seed_times=122)
    df_arrivals = pd.DataFrame({
        "Job": list(new_jssp_data.keys()),
        "Arrival": new_arrival_times
    })

    # Filtern
    df_arrivals = df_arrivals[
        (df_arrivals["Arrival"] >= new_day_start) &
        (df_arrivals["Arrival"] < new_day_start + 1440)
    ]

    # 5) new_jssp_data auf die tatsächlich verbleibenden Jobs kürzen
    valid_job_ids = set(df_arrivals["Job"])
    new_jssp_data = {job_id: ops for job_id, ops in new_jssp_data.items() if job_id in valid_job_ids}

    if generate_deadlines:
        k_opt, deadlines = sdead.find_k(new_jssp_data, df_arrivals, ssv.schedule_fcfs_with_arrivals, target_service=1, buffer_factor = deadlines_buffer_factor)
        df_arrivals_deadlines = df_arrivals.assign(Deadline=df_arrivals["Job"].map(deadlines)).sort_values("Arrival")
        return new_jssp_data, df_arrivals_deadlines

    return new_jssp_data, df_arrivals


# ------------------------------------------------------------------------------------------------------

def create_new_jobs(job_set_dict: dict, new_instance: dict, shuffle: bool = False, seed: int = 50):
    """
    Erzeugt neue Jobs mit fortlaufenden IDs. 
    Der Offset ist die nächsthöhere Nummer nach der größten vorhandenen Job-ID,
    nicht-matching IDs bekommen einen Suffix von 0.
    """
    # 1) Items aus dem new_instance-Dict auslesen
    items = list(new_instance.items())

    # 2) Optional mischen
    if shuffle:
        random.seed(seed)
        random.shuffle(items)

    # 3) Offset ermitteln: höchste vorhandene Nummer + 1,
    #    nicht-matching Job-Keys zählen als 0
    suffixes = []
    for job_id in (job_set_dict or {}):
        m = re.search(r"Job_(\d+)$", job_id)
        suffixes.append(int(m.group(1)) if m else 0)

    offset = (max(suffixes) + 1) if suffixes else 0

    # 4) Neues Dict zusammenbauen
    new_jobs = {}
    for i, (_, ops) in enumerate(items):
        job_name = f"Job_{offset + i:03d}"
        new_jobs[job_name] = ops

    return new_jobs


def create_new_arrivals(numb_jobs: int, mean_interarrival_time: float, start_time: int = 0, random_seed_times: int = 122):
    # 1) Seed setzen für Reproduzierbarkeit
    np.random.seed(random_seed_times)

    # 2) Interarrival-Zeiten erzeugen
    interarrival_times = np.random.exponential(scale=mean_interarrival_time, size=numb_jobs)

    # 3) Kumulieren ab last_arrival und auf 2 Nachkommastellen runden
    new_arrivals = np.round(start_time + np.cumsum(interarrival_times), 2)

    return new_arrivals