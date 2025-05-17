import re
import random
import numpy as np
import pandas as pd

import utils.simple.schedule_solver as ssv
import utils.deadline.generator as deadline_gen
import utils.schedule_interarrival as sit



# ------------------------------------------------------------------------------------------------------


def init_jobs_with_arrivals(df_template: pd.DataFrame,
                            day_num: int,
                            u_b_mmax: float = 0.9,
                            generate_deadlines: bool = False,
                            deadlines_buffer_factor: float = 2.0
                           ) -> (pd.DataFrame, pd.DataFrame):
    """
    Erzeugt über `day_num` Tage neue Jobs (DataFrame) mit Ankünften.

    Rückgabe:
    - df_all_jobs: DataFrame aller erzeugten Jobs (je Zeile eine Operation)
    - df_all_arrivals: DataFrame aller Ankünfte mit Spalten ['Job','Arrival'] (ggf. Deadline)
    """
    df_combined_jobs = pd.DataFrame(columns=df_template.columns)
    df_arrs_list     = []
    df_old_jobs      = pd.DataFrame(columns=df_template.columns)
    df_old_arrivals  = pd.DataFrame(columns=['Job','Arrival'])

    for _ in range(day_num):
        # Generiere für einen Tag
        df_new_jobs, df_arrivals = create_new_jobs_with_arrivals_for_one_day(
            df_old_jobs,
            df_old_arrivals,
            df_jssp=df_template,
            u_b_mmax=u_b_mmax,
            shuffle=False,
            random_seed_jobs=0,
            random_seed_times=0,
            generate_deadlines=generate_deadlines,
            deadlines_buffer_factor=deadlines_buffer_factor
        )
        # Jobs anhängen
        df_combined_jobs = pd.concat([df_combined_jobs, df_new_jobs], ignore_index=True)
        # Ankünfte sammeln
        df_arrs_list.append(df_arrivals)
        # Basis für nächsten Tag setzen
        df_old_jobs     = df_new_jobs
        df_old_arrivals = df_arrivals

    df_all_arrivals = pd.concat(df_arrs_list, ignore_index=True)
    return df_combined_jobs, df_all_arrivals



def update_new_day(df_existing_jobs: pd.DataFrame,
                   df_existing_arrivals: pd.DataFrame,
                   df_jssp: pd.DataFrame,
                   u_b_mmax: float = 0.9,
                   shuffle: bool = False,
                   generate_deadlines: bool = False,
                   deadlines_buffer_factor: float = 2.0
                  ) -> (pd.DataFrame, pd.DataFrame):
    """
    Hängt für einen weiteren Tag Jobs und Ankünfte an die bestehenden DataFrames an.
    """
    df_new_jobs, df_new_arrivals = create_new_jobs_with_arrivals_for_one_day(
        df_existing_jobs,
        df_existing_arrivals,
        df_jssp,
        u_b_mmax=u_b_mmax,
        shuffle=shuffle,
        random_seed_jobs=0,
        random_seed_times=0,
        generate_deadlines=generate_deadlines,
        deadlines_buffer_factor=deadlines_buffer_factor
    )

    df_jobs     = pd.concat([df_existing_jobs, df_new_jobs],     ignore_index=True)
    df_arrivals = pd.concat([df_existing_arrivals, df_new_arrivals], ignore_index=True)

    return df_jobs.reset_index(drop=True), df_arrivals.reset_index(drop=True)



# ------------------------------------------------------------------------------------------------------

def create_new_jobs_with_arrivals_for_one_day(df_old_jobs: pd.DataFrame,
                                             df_old_arrivals: pd.DataFrame,
                                             df_jssp: pd.DataFrame,
                                             u_b_mmax: float = 0.9,
                                             shuffle: bool = False,
                                             random_seed_jobs: int = 50,
                                             random_seed_times: int = 122,
                                             generate_deadlines: bool = False,
                                             deadlines_buffer_factor: float = 2.0
                                            ) -> (pd.DataFrame, pd.DataFrame):
    # 0) Leere-DF-Fallback
    if df_old_jobs is None:
        df_old_jobs = pd.DataFrame(columns=df_jssp.columns)
    if df_old_arrivals is None:
        df_old_arrivals = pd.DataFrame(columns=['Job','Arrival'])

    # 1) Tagesstart
    if not df_old_arrivals.empty:
        last = df_old_arrivals['Arrival'].max()
        day_start = ((last // 1440) + 1) * 1440
    else:
        day_start = 0

    # 2) Instanz vervielfachen: dreimal hintereinander neue Jobs aus df_jssp
    df_prev = df_old_jobs.copy()
    df_all_new = pd.DataFrame(columns=df_jssp.columns)
    for i in range(3):
        flag = shuffle if (i % 2 == 0) else not shuffle
        df_new = create_new_jobs(df_prev, df_jssp, shuffle=flag, seed=random_seed_jobs)
        if df_all_new.empty:
            df_all_new = df_new.copy()
        else:
            df_all_new = pd.concat([df_all_new, df_new], ignore_index=True)
        df_prev = df_new

    # 3) mittlere Interarrival
    t_a = sit.calculate_mean_interarrival_time(df_all_new, u_b_mmax=u_b_mmax)

    # 4) Ankunftszeiten
    df_arr = create_new_arrivals(df_all_new,
                                 mean_interarrival_time=t_a,
                                 start_time=day_start,
                                 random_seed=random_seed_times)

    # 5) nur aktueller Tag
    df_arr = df_arr[
        (df_arr['Arrival'] >= day_start) &
        (df_arr['Arrival'] < day_start + 1440)
    ].reset_index(drop=True)

    # 6) verbleibende Jobs
    valid = set(df_arr['Job'])
    df_jobs = df_all_new[df_all_new['Job'].isin(valid)].reset_index(drop=True)

    # 7) Deadlines optional
    if generate_deadlines:
        k_opt, df_dead = deadline_gen.find_k(
            df_jobs, df_arr, ssv.schedule_fcfs_with_arrivals,
            target_service=1.0, buffer_factor=deadlines_buffer_factor
        )
        df_arr = df_arr.merge(df_dead, on='Job', how='left') \
                       .sort_values('Arrival') \
                       .reset_index(drop=True)

    return df_jobs, df_arr


# ------------------------------------------------------------------------------------------------------


def create_new_arrivals(df_jobs: pd.DataFrame,
                        mean_interarrival_time: float,
                        start_time: float = 0.0,
                        random_seed: int = 122) -> pd.DataFrame:
    # 1) Seed setzen für Reproduzierbarkeit
    np.random.seed(random_seed)

    # 2) Interarrival-Zeiten erzeugen
    jobs = df_jobs['Job'].unique().tolist()
    interarrival_times = np.random.exponential(scale=mean_interarrival_time, size=len(jobs))
    # interarrival_times[0] = 0.0  # Start bei 0 Minuten
    interarrival_times[0] = 0.0

    # 3) Kumulieren ab start_time und auf 2 Nachkommastellen runden
    new_arrivals = np.round(start_time + np.cumsum(interarrival_times), 2)

    return pd.DataFrame({
        'Job': jobs,
        'Arrival': new_arrivals
    })



def create_new_jobs(df_existing: pd.DataFrame,
                    df_template: pd.DataFrame,
                    shuffle: bool = False,
                    seed: int = 50) -> pd.DataFrame:
    """
    Erzeugt aus df_template neue Jobs mit fortlaufenden IDs.
    Liefert nur die neuen Jobs, nicht bestehende.

    - df_existing: DataFrame mit Spalte 'Job' im Format 'Job_XXX'
    - df_template: DataFrame mit ['Job','Operation','Machine','Processing Time']
    - shuffle: optionales Mischen der Template-Jobs
    - seed: RNG-Seed für’s Mischen
    """
    # 1) Offset ermitteln
    if df_existing is None or df_existing.empty:
        offset = 0
    else:
        nums = (
            df_existing['Job']
            .str.extract(r'Job_(\d+)$')[0]
            .dropna()
            .astype(int)
        )
        offset = nums.max() + 1 if not nums.empty else 0

    # 2) Template-Job-Gruppen (je ursprünglichem Job ein Block)
    groups = [grp for _, grp in df_template.groupby('Job', sort=False)]

    # 3) Optional mischen
    if shuffle:
        random.seed(seed)
        random.shuffle(groups)

    # 4) Neue Jobs erzeugen
    new_recs = []
    for i, grp in enumerate(groups):
        new_id = f"Job_{offset + i:03d}"
        for _, row in grp.iterrows():
            new_recs.append({
                'Job': new_id,
                'Operation': row['Operation'],
                'Machine': row['Machine'],
                'Processing Time': row['Processing Time']
            })

    # 5) Nur die neuen Jobs zurückgeben
    return pd.DataFrame(new_recs).reset_index(drop=True)
