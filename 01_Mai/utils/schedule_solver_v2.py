import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple



# Dispatching Rules ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------

# FCFS schedules operations by their earliest possible start time, breaking ties in favor of the job that arrived first
def schedule_fcfs_with_arrivals(jobs: Dict[int, List[Tuple[int, float]]],
                                arrival_df: pd.DataFrame) -> pd.DataFrame:
    """
    FCFS-Scheduling mit Job-Ankunftszeiten.
    """
    # --- Vorbereitungen -----------------------------------------------------
    arrival_times = (
        arrival_df.set_index("Job")["Arrival"]
        .to_dict()
    )

    # Zustand der Jobs und Maschinen
    next_op_idx  = {job: 0                       for job in jobs}               # nächste Op-Nr.
    job_ready    = {job: arrival_times[job]      for job in jobs}               # frühester Start
    machine_ready = defaultdict(float)                                              # frei ab
    remaining_ops = sum(len(ops) for ops in jobs.values())

    schedule = []

    # --- Hauptschleife ------------------------------------------------------
    while remaining_ops:
        chosen = None         # (job, machine, duration, earliest_start)

        # Suche beste nächste Operation (globale FCFS-Logik)
        for job, idx in next_op_idx.items():
            if idx >= len(jobs[job]):
                continue   # Job fertig

            machine, duration = jobs[job][idx]
            earliest_start    = max(job_ready[job], machine_ready[machine])

            if (chosen is None or
                earliest_start < chosen[3] or
                (earliest_start == chosen[3] and arrival_times[job] < arrival_times[chosen[0]])):
                chosen = (job, machine, duration, earliest_start)

        # Plane die gewählte Operation
        job, machine, duration, start = chosen
        end = start + duration
        schedule.append(
            {
                "Job":      f"{job}",
                "Machine":  f"M{machine}",
                "Start":    start,
                "Processing Time": duration,
                "End":      end,
            }
        )

        # Zustände aktualisieren
        job_ready[job]         = end
        machine_ready[machine] = end
        next_op_idx[job]      += 1
        remaining_ops         -= 1

    df_schedule = pd.DataFrame(schedule)
    df_schedule["Arrival"] = df_schedule["Job"].map(arrival_times)
    df_schedule = df_schedule[["Job", "Arrival", "Machine", "Start", "Processing Time", "End"]]

    return df_schedule.sort_values(by=["Arrival", "Start"])