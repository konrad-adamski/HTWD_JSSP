import pandas as pd                               # Für DataFrame-Erzeugung und -Manipulation
from collections import defaultdict               # Für Dictionary mit Default-Werten (Maschinenverfügbarkeit)
from typing import Dict, List, Tuple              # Typ-Hinweise: Dict, List, Tuple

import pulp
# conda install -c conda-forge highs


# FCFS schedules operations by their earliest possible start time ------------
def schedule_fcfs(jobs: Dict[int, List[Tuple[int, float]]]) -> pd.DataFrame:
    """
    FCFS-Scheduling ohne Job-Ankunftszeiten.
    Alle Jobs stehen ab Zeit 0 zur Verfügung,  
    Ties werden nach numerischer Job-ID aufgelöst.
    """
    # --- Vorbereitungen -----------------------------------------------------
    # job_ready[j] = früheste Zeit, zu der Job j seine nächste Operation starten kann
    job_ready     = {job: 0.0 for job in jobs}
    # machine_ready[m] = früheste Zeit, zu der Maschine m wieder frei ist
    machine_ready = defaultdict(float)
    # next_op_idx[j] = Index der nächsten auszuführenden Operation in jobs[j]
    next_op_idx   = {job: 0 for job in jobs}
    # verbleibende Gesamtzahl aller Operationen
    remaining_ops = sum(len(ops) for ops in jobs.values())

    # Hier sammeln wir später alle geplanten Operationseinträge
    schedule = []

    # --- Hauptschleife ------------------------------------------------------
    # Solange noch Operationen übrig sind, wiederhole Planungsschritt
    while remaining_ops > 0:
        chosen = None  # Wird Tuple (job, machine, duration, earliest_start) enthalten

        # Durchlaufe alle Jobs, um die nächste FCFS-Operation zu finden
        for job, idx in next_op_idx.items():
            # Wenn alle Operationen dieses Jobs bereits geplant sind, überspringen
            if idx >= len(jobs[job]):
                continue

            machine, duration = jobs[job][idx]
            # frühester Startzeitpunkt = max(Auftragsbereitschaft, Maschinenbereitschaft)
            earliest_start = max(job_ready[job], machine_ready[machine])

            # Auswahlkriterium:
            # 1) Wenn noch keine Wahl getroffen (chosen is None)
            # 2) Wenn dieser Start früher ist als der aktuell Gewählte
            # 3) Bei Gleichstand: Job mit kleinerer ID gewinnt
            if (chosen is None
                or earliest_start < chosen[3]
                or (earliest_start == chosen[3] and job < chosen[0])
               ):
                chosen = (job, machine, duration, earliest_start)

        # --- Planung der gewählten Operation ---
        job, machine, duration, start = chosen
        end = start + duration

        # Operation in die Schedule-Liste aufnehmen
        schedule.append({
            "Job":             job,
            "Machine":         f"M{machine}",
            "Start":           start,
            "Processing Time": duration,
            "End":             end,
        })

        # --- Zustandsaktualisierung nach Planung ---
        # Job kann erst nach Ende dieser Operation seine nächste Op starten
        job_ready[job]         = end
        # Maschine ist bis Ende der Operation belegt
        machine_ready[machine] = end
        # Für den Job die Index auf die nächste Operation weiterschalten
        next_op_idx[job]      += 1
        # Eine Operation weniger zu planen
        remaining_ops         -= 1

    # --- Ergebniszusammenstellung ---
    df = pd.DataFrame(schedule)
    # Sortierung: zuerst nach Startzeit, dann Job-ID (nur für deterministische Reihenfolge)
    return df.sort_values(by=["Start", "Job"]).reset_index(drop=True)
