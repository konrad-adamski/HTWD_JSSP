import pandas as pd

def get_jssp_from_schedule(df_schedule: pd.DataFrame, duration_column: str = "Processing Time") -> dict:
    job_dict = {}

    df_schedule = df_schedule.copy()
    df_schedule["Machine"] = df_schedule["Machine"].str.extract(r"M(\d+)").astype(int)
    df_schedule[duration_column] = df_schedule[duration_column].astype(int)

    for job, machine, duration in zip(df_schedule["Job"], df_schedule["Machine"], df_schedule[duration_column]):
        if job not in job_dict:
            job_dict[job] = []
        job_dict[job].append([machine, duration])

    return job_dict