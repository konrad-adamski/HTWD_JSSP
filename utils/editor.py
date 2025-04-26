import pandas as pd


# Dataframe der Operation die erst nach gegebenen Zeitlimit abgeschlossen werden
def get_operations_ending_after_day_limit(df: pd.DataFrame, daily_timelimit_h: float) -> pd.DataFrame:
    daily_timelimit = daily_timelimit_h * 60  # Umrechnung in Minuten
    return df[df["End"] > daily_timelimit].copy()

def separate_operation_by_day_limit(df_schedule: pd.DataFrame, day_limit_h: float = 21):
    """
    Trennt einen Operationsplan in:
    - df_schedule_on_time: Operationen, die innerhalb des Tageslimits enden
    - df_late: Operationen, die nach dem Tageslimit enden

    :param df_schedule: DataFrame mit Spalte "End" (Endzeit in Minuten)
    :param day_limit_h: Tageszeitlimit in Stunden (Standard: 21h)
    :return: df_schedule_on_time, df_late
    """
    day_limit_minutes = day_limit_h * 60  # Umrechnung Stunden → Minuten

    df_schedule_on_time = df_schedule[df_schedule["End"] <= day_limit_minutes].copy()
    df_late = df_schedule[df_schedule["End"] > day_limit_minutes].copy()

    return df_schedule_on_time, df_late



def get_jssp_from_schedule(df_schedule: pd.DataFrame) -> dict:
    job_dict = {}

    df_schedule = df_schedule.copy()
    df_schedule["Machine"] = df_schedule["Machine"].str.extract(r"M(\d+)").astype(int)
    df_schedule["Duration"] = df_schedule["Duration"].astype(int)

    for job_name, group in df_schedule.groupby("Job"):
        group_sorted = group.sort_values("Start")  # technologische Reihenfolge erhalten
        operations = group_sorted[["Machine", "Duration"]].values.tolist()

        job_dict[job_name] = operations

    return job_dict

    
# Merging/Adding Jobs -------------------------------------------------------------------------------------------------------
def merge_jssp_jobs(new_jobs: dict, remained_jobs: dict) -> dict:
    # Kombiniere beide Dictionaries
    merged = {**remained_jobs, **new_jobs}
    return merged 


def merge_jobs(jobs_a, jobs_b):
    """
    Kombiniert zwei Job-Dictionaries.
    Falls eines None ist, wird das andere übernommen.
    """
    jobs_a = jobs_a or {}
    jobs_b = jobs_b or {}
    return {**jobs_a, **jobs_b}


def add_remaining_jobs_with_zero_arrival(df_arrivals_new: pd.DataFrame, remaining_jobs: dict, day_id: int) -> pd.DataFrame:
    job_names = list(remaining_jobs.keys())

    df_remaining = pd.DataFrame({
        "Job-ID": job_names,
        "Day-ID": [day_id] * len(job_names),
        "Ankunftszeit (Minuten)": [0.0] * len(job_names)
    })
    
    return pd.concat([df_remaining, df_arrivals_new], ignore_index=True)