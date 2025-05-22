import re
import random
import numpy as np
import pandas as pd

import utils.simple.schedule_solver__arrival as ssv
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


# Jobs Sample Beforehand -------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------

def add_beforehand_jobs_to_current_horizon(df_existing_jobs: pd.DataFrame,
                                           df_existing_times: pd.DataFrame,
                                           df_jssp: pd.DataFrame,
                                           df_times: pd.DataFrame,
                                           min_arrival_time: float,
                                           n: int = 3,
                                           random_state: int = None):
    """
    Ergänzt bestehende Jobs durch `n` zusätzliche Jobs mit Arrival-Zeit >= `min_arrival_time`.
    Verwendet intern `sample_jobs_with_times_after_arrivaltime`, schließt aber bereits geplante Jobs aus.

    Parameter:
    - df_existing_jobs (pd.DataFrame): Bereits geplante JSSP-Daten
    - df_existing_times (pd.DataFrame): Bereits geplante Zeitinformationen
    - df_jssp (pd.DataFrame): Vollständiger JSSP-DataFrame mit Spalte 'Job'
    - df_times (pd.DataFrame): Vollständiger Zeit-DataFrame mit Spalten 'Job' und 'Arrival'
    - min_arrival_time (float): Untere Zeitgrenze für die Auswahl
    - n (int): Anzahl zusätzlicher Jobs (Standard: 3)
    - random_state (int oder None): Seed für Reproduzierbarkeit (optional)

    Rückgabe:
    - Tuple von zwei kombinierten DataFrames:
        (1) df_combined_jobs: erweiterte JSSP-Daten
        (2) df_combined_times: erweiterte Zeitinformationen
    """

    # Nur Jobs berücksichtigen, die noch nicht enthalten sind
    remaining_times = df_times[~df_times["Job"].isin(df_existing_times["Job"])].copy()

    # Verwende die bestehende Sampling-Funktion
    df_sampled_jobs, df_sampled_times = sample_jobs_with_times_after_arrivaltime(
        df_jssp=df_jssp,
        df_times=remaining_times,
        min_arrival_time=min_arrival_time,
        n=n,
        random_state=random_state
    )

    # Kombinieren mit bestehenden Daten
    df_combined_jobs = pd.concat([df_existing_jobs, df_sampled_jobs], ignore_index=True)
    df_combined_times = pd.concat([df_existing_times, df_sampled_times], ignore_index=True)

    return df_combined_jobs, df_combined_times



def sample_jobs_with_times_after_arrivaltime(df_jssp, df_times, min_arrival_time, n=3, random_state=None):
    """
    Wählt zufällig `n` Jobs aus, deren Ankunftszeit nach `min_arrival_time` liegt,
    und gibt sowohl deren JSSP-Daten als auch deren Zeitinformationen zurück.

    Parameter:
    - df_jssp (pd.DataFrame): JSSP-DataFrame mit einer Spalte 'Job'
    - df_times (pd.DataFrame): DataFrame mit mindestens den Spalten 'Job' und 'Arrival'
    - min_arrival_time (float): Untere Zeitgrenze für die Auswahl
    - n (int): Anzahl der zufällig auszuwählenden Jobs (Standard: 3)
    - random_state (int oder None): Seed für Reproduzierbarkeit (optional)

    Rückgabe:
    - Tuple von zwei DataFrames:
        (1) df_sampled_jobs: vollständige JSSP-Daten der ausgewählten Jobs
        (2) df_sampled_times: Zeitdaten der ausgewählten Jobs
    """
    time_filter = df_times["Arrival"] >= min_arrival_time
    df_times_filtered = df_times[time_filter].copy()

    if len(df_times_filtered) >= n:
        df_sampled_times = df_times_filtered.sample(n=n, random_state=random_state)
        df_sampled_jobs = df_jssp[df_jssp["Job"].isin(df_sampled_times["Job"])]
    else:
        df_sampled_times = pd.DataFrame(columns=df_times.columns)
        df_sampled_jobs = pd.DataFrame(columns=df_jssp.columns)

    return df_sampled_jobs, df_sampled_times

# Extra Jobs -------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------


def add_extra_jobs_to_current_horizon(df_existing_jobs: pd.DataFrame,
                       df_existing_arrivals: pd.DataFrame,
                       df_template: pd.DataFrame,
                       start_time: float,
                       start_index: int,
                       job_prefix: str = "",
                       mean_interarrival_time: float = 120,
                       job_numb: int = 10,
                       shuffle: bool = False,
                       seed_jobs: int = 42,
                       seed_arrivals: int = 122
                      ) -> (pd.DataFrame, pd.DataFrame):
    """
    Fügt neue Jobs + Ankünfte ab `start_time` hinzu, basierend auf einem gegebenen mean_interarrival_time.
    Nutzt `create_new_jobs_with_arrivals_with_prefix`.
    """

    # 1. Neue Jobs und Ankünfte erzeugen
    df_new_jobs, df_new_arrivals = create_new_jobs_with_arrivals_with_prefix(
        df_template=df_template,
        job_numb=job_numb,
        job_prefix=job_prefix,
        start_index=start_index,
        start_time=start_time,
        mean_interarrival_time=mean_interarrival_time,
        shuffle=shuffle,
        seed_jobs=seed_jobs,
        seed_arrivals=seed_arrivals
    )

    # 2. Anfügen und sortieren
    df_arrivals = pd.concat([df_existing_arrivals, df_new_arrivals], ignore_index=True)
    df_arrivals = df_arrivals.sort_values('Arrival').reset_index(drop=True)

    df_jobs = pd.concat([df_existing_jobs, df_new_jobs], ignore_index=True)
    job_order = pd.Categorical(df_jobs['Job'], categories=df_arrivals['Job'], ordered=True)
    df_jobs = df_jobs.assign(_order=job_order) \
                     .sort_values(by=['_order', 'Operation']) \
                     .drop(columns=['_order']) \
                     .reset_index(drop=True)

    return df_jobs, df_arrivals





def create_new_jobs_with_arrivals_with_prefix(df_template: pd.DataFrame,
                                              job_numb: int,
                                              job_prefix: str = "",
                                              start_index: int = 0,
                                              start_time: float = 0.0,
                                              mean_interarrival_time: float = 50.0,
                                              shuffle: bool = False,
                                              seed_jobs: int = 42,
                                              seed_arrivals: int = 122
                                             ) -> (pd.DataFrame, pd.DataFrame):
    """
    Erstellt neue Jobs (mit Prefix) und passende Ankunftszeiten ab `start_time`.

    Parameter:
    - df_template: Vorlage mit ['Job','Operation','Machine','Processing Time']
    - job_numb: Anzahl neuer Jobs
    - job_prefix: z. B. 'A' → Job_A000, Job_A001, ...
    - start_index: Startindex für die Nummerierung
    - start_time: Zeitpunkt, ab dem neue Ankünfte beginnen sollen
    - mean_interarrival_time: Mittelwert (λ) der Exponentialverteilung
    - shuffle: Optional zufällige Reihenfolge der Vorlagejobs
    - seed_jobs: Seed für Job-Mischung
    - seed_arrivals: Seed für Arrival-Erzeugung

    Rückgabe:
    - df_jobs: DataFrame mit Job-Operationen
    - df_arrivals: DataFrame mit ['Job', 'Arrival']
    """
    # 1. Neue Jobs erzeugen
    df_jobs = create_new_jobs_with_prefix(
        df_template=df_template,
        job_numb=job_numb,
        job_prefix=job_prefix,
        start_index=start_index,
        shuffle=shuffle,
        seed=seed_jobs
    )

    # 2. Arrival-Zeiten erzeugen
    df_arrivals = create_new_arrivals(
        df_jobs,
        mean_interarrival_time=mean_interarrival_time,
        start_time=start_time,
        random_seed=seed_arrivals
    )

    return df_jobs, df_arrivals



def create_new_jobs_with_prefix(df_template: pd.DataFrame,
                                job_numb: int,
                                job_prefix: str = "",
                                start_index: int = 0,
                                shuffle: bool = False,
                                seed: int = 42) -> pd.DataFrame:
    """
    Erzeugt `job_numb` neue Jobs aus dem Template mit gewünschtem Prefix (z. B. 'A' → Job_A000).
    
    Parameter:
    - df_template: DataFrame mit ['Job','Operation','Machine','Processing Time']
    - job_numb: Anzahl neuer Jobs (Blöcke aus df_template)
    - job_prefix: Optionaler Präfix für Job-IDs (z. B. 'A' → Job_A001)
    - start_index: Startindex für Nummerierung (z. B. 0 oder 100)
    - shuffle: Ob die Reihenfolge der Vorlage-Jobs gemischt werden soll
    - seed: RNG-Seed fürs Mischen

    Rückgabe:
    - DataFrame mit den neuen Jobs
    """
    # Vorlage-Gruppen (je ein Job-Block)
    job_groups = [grp for _, grp in df_template.groupby('Job', sort=False)]

    if shuffle:
        random.seed(seed)
        random.shuffle(job_groups)

    new_jobs = []
    for i in range(job_numb):
        grp = job_groups[i % len(job_groups)]  # zyklisch durch Vorlage
        job_id = f"Job_{job_prefix}{start_index + i:03d}"
        for _, row in grp.iterrows():
            new_jobs.append({
                'Job': job_id,
                'Operation': row['Operation'],
                'Machine': row['Machine'],
                'Processing Time': row['Processing Time']
            })

    return pd.DataFrame(new_jobs).reset_index(drop=True)

