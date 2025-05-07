import pulp
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple
# conda install -c conda-forge highs

# Exact MILP Formulation ------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------

# MILP with HiGHS - minimizing all FlowTimes -----------------------------------------------------------------------
def solve_jssp_individual_flowtime(job_dict, df_arrivals, solver_time_limit=300, epsilon=0.00):
    """
    Minimiert die Summe der individuellen Durchlaufzeiten (Flow Times) aller Jobs.
    Zielfunktion: sum_j [ Endzeit_j - Ankunftszeit_j ]

    Parameter:
    - job_dict: Dictionary mit Jobdaten (jede Operation als (Maschine, Dauer))
    - df_arrivals: DataFrame mit Spalten "Job" und "Arrival"
    - solver_time_limit: Max. Zeit in Sekunden für den Solver
    - epsilon: Kleiner Abstand zur Vermeidung von Maschinenkonflikten
    """

    # Ankunftszeiten als Dictionary
    df_arrivals = df_arrivals.sort_values("Arrival").reset_index(drop=True)
    arrival_times = df_arrivals.set_index("Job")["Arrival"].to_dict()

    # Jobnamen nach Ankunftszeit sortieren (absteigend)
    job_names = list(df_arrivals.sort_values("Arrival", ascending=False)["Job"])

    num_jobs = len(job_names)
    all_ops = [job_dict[job] for job in job_names]
    all_machines = {op[0] for job in all_ops for op in job}

    # LP-Problem
    prob = pulp.LpProblem("JobShop_Total_FlowTime", pulp.LpMinimize)

    # Variablen
    starts = {
        (j, o): pulp.LpVariable(f"start_{j}_{o}", lowBound=0, cat="Continuous")
        for j in range(num_jobs) for o in range(len(all_ops[j]))
    }

    job_ends = {
        j: pulp.LpVariable(f"job_end_{j}", lowBound=0, cat="Continuous")
        for j in range(num_jobs)
    }

    # Zielfunktion: ungeeichtete Summe aller Flow Times
    prob += pulp.lpSum([
        job_ends[j] - arrival_times[job_names[j]]
        for j in range(num_jobs)
    ])

    # Technologische Reihenfolge & Ankunftszeit
    for j, job_name in enumerate(job_names):
        job = job_dict[job_name]
        prob += starts[(j, 0)] >= arrival_times[job_name]
        for o in range(1, len(job)):
            d_prev = job[o - 1][1]
            prob += starts[(j, o)] >= starts[(j, o - 1)] + d_prev

    # Maschinenkonflikte (Disjunktivität)
    bigM = 1e5
    for m in all_machines:
        ops = [(j, o, d) for j in range(num_jobs)
               for o, (mach, d) in enumerate(all_ops[j]) if mach == m]
        for i in range(len(ops)):
            j1, o1, d1 = ops[i]
            for j2, o2, d2 in ops[i + 1:]:
                if j1 != j2:
                    y = pulp.LpVariable(f"y_{j1}_{o1}_{j2}_{o2}", cat="Binary")
                    prob += starts[(j1, o1)] + d1 + epsilon <= starts[(j2, o2)] + bigM * (1 - y)
                    prob += starts[(j2, o2)] + d2 + epsilon <= starts[(j1, o1)] + bigM * y

    # Endzeit jeder Job = Ende letzter Operation
    for j in range(num_jobs):
        last_op = len(all_ops[j]) - 1
        prob += job_ends[j] >= starts[(j, last_op)] + all_ops[j][last_op][1]

    # Solver starten
    solver = pulp.HiGHS_CMD(msg=True, timeLimit=solver_time_limit)
    prob.solve(solver)

    # Zeitplan extrahieren
    schedule_data = []
    for (j, o), var in sorted(starts.items()):
        start = var.varValue
        if start is not None:
            machine, duration = all_ops[j][o]
            end = start + duration
            schedule_data.append({
                "Job": job_names[j],
                "Machine": f"M{machine}",
                "Start": round(start, 2),
                "Processing Time": duration,
                "End": round(end, 2)
            })

    df_schedule = pd.DataFrame(schedule_data)
    df_schedule["Arrival"] = df_schedule["Job"].map(arrival_times)
    df_schedule["Flow time"] = df_schedule["End"] - df_schedule["Arrival"]

    df_schedule = df_schedule[["Job", "Arrival", "Machine", "Start", "Processing Time", "Flow time", "End"]]

    total_flowtime = round(pulp.value(prob.objective), 3)
    return df_schedule, total_flowtime



# MILP with HiGHS - minimizing all FlowTimes (weightend) ------------------------------------------------------------
def solve_jssp_weighted_individual_flowtime(job_dict, df_arrivals, solver_time_limit=300, epsilon=0.00):
    """
    Minimiert die gewichtete Summe der individuellen Durchlaufzeiten (Flow Times) aller Jobs.
    Gewichtung bevorzugt früh ankommende Jobs.

    Gewicht_j = 1 / (1 + Ankunftszeit_j)
    Zielfunktion: sum_j [ Gewicht_j * (Endzeit_j - Ankunftszeit_j) ]

    Parameter:
    - job_dict: Dictionary mit Jobdaten (jede Operation als (Maschine, Dauer))
    - df_arrivals: DataFrame mit Spalten "Job" und "Arrival"
    - solver_time_limit: Max. Zeit in Sekunden für den Solver
    - epsilon: Kleiner Abstand zur Vermeidung von Maschinenkonflikten
    """

    # Ankunftszeiten als Dictionary
    df_arrivals = df_arrivals.sort_values("Arrival").reset_index(drop=True)
    arrival_times = df_arrivals.set_index("Job")["Arrival"].to_dict()

    # Jobnamen nach Ankunftszeit sortieren (absteigend)
    job_names = list(df_arrivals.sort_values("Arrival", ascending=False)["Job"])

    num_jobs = len(job_names)

    # Operationen in Ankunftsreihenfolge
    all_ops = [job_dict[job] for job in job_names]

    # Maschinen extrahieren
    all_machines = {op[0] for job in all_ops for op in job}

    # LP-Problem
    prob = pulp.LpProblem("JobShop_Weighted_Total_FlowTime", pulp.LpMinimize)

    # Startzeitvariablen
    starts = {
        (j, o): pulp.LpVariable(f"start_{j}_{o}", lowBound=0, cat="Continuous")
        for j in range(num_jobs) for o in range(len(all_ops[j]))
    }

    # Endzeitvariablen je Job
    job_ends = {
        j: pulp.LpVariable(f"job_end_{j}", lowBound=0, cat="Continuous")
        for j in range(num_jobs)
    }

    # Gewichtung: frühere Ankunft = höhere Priorität
    weights = {j: 1 / (1 + arrival_times[job_names[j]]) for j in range(num_jobs)}

    # Zielfunktion: gewichtete Durchlaufzeiten minimieren
    prob += pulp.lpSum([
        weights[j] * (job_ends[j] - arrival_times[job_names[j]])
        for j in range(num_jobs)
    ])

    # Technologische Reihenfolge + Ankunft
    for j, job_name in enumerate(job_names):
        job = job_dict[job_name]
        prob += starts[(j, 0)] >= arrival_times[job_name]
        for o in range(1, len(job)):
            d_prev = job[o - 1][1]
            prob += starts[(j, o)] >= starts[(j, o - 1)] + d_prev

    # Maschinenkonflikte
    bigM = 1e5
    for m in all_machines:
        ops = [(j, o, d) for j in range(num_jobs)
               for o, (mach, d) in enumerate(all_ops[j]) if mach == m]
        for i in range(len(ops)):
            j1, o1, d1 = ops[i]
            for j2, o2, d2 in ops[i + 1:]:
                if j1 != j2:
                    y = pulp.LpVariable(f"y_{j1}_{o1}_{j2}_{o2}", cat="Binary")
                    prob += starts[(j1, o1)] + d1 + epsilon <= starts[(j2, o2)] + bigM * (1 - y)
                    prob += starts[(j2, o2)] + d2 + epsilon <= starts[(j1, o1)] + bigM * y

    # Endzeit je Job = Ende letzter Operation
    for j in range(num_jobs):
        last_op = len(all_ops[j]) - 1
        prob += job_ends[j] >= starts[(j, last_op)] + all_ops[j][last_op][1]

    # Solver
    solver = pulp.HiGHS_CMD(msg=True, timeLimit=solver_time_limit)
    prob.solve(solver)

    # Ergebnisse extrahieren
    schedule_data = []
    for (j, o), var in sorted(starts.items()):
        start = var.varValue
        if start is not None:
            machine, duration = all_ops[j][o]
            end = start + duration
            schedule_data.append({
                "Job": job_names[j],
                "Machine": f"M{machine}",
                "Start": round(start, 2),
                "Processing Time": duration,
                "End": round(end, 2)
            })

    df_schedule = pd.DataFrame(schedule_data)

    df_schedule["Arrival"] = df_schedule["Job"].map(arrival_times)
    df_schedule["Flow time"] = df_schedule["End"] - df_schedule["Arrival"]

    df_schedule = df_schedule[["Job", "Arrival", "Machine", "Start", "Processing Time", "Flow time", "End"]]

    # Gesamtzielwert
    total_weighted_flowtime = round(pulp.value(prob.objective), 3)

    return df_schedule, total_weighted_flowtime



# MILP with HiGHS - minimizing the entire Makespan -----------------------------------------------------------------
def solve_jssp_global_makespan(job_dict, df_arrivals, solver_time_limit=300, epsilon=0.06):
    """
    Erste Stufe: Minimierung des Makespan (Gesamtdauer) eines Job-Shop-Problems.

    Parameter:
    - epsilon: Kleiner Sicherheitsabstand (in Minuten) zwischen Operationen auf derselben Maschine,
               um numerische Ungenauigkeiten und Maschinenkonflikte zu vermeiden (z.B. 0.06 Minuten = 3.6 Sekunden).
    """


    # Ankunftszeiten als Dictionary
    df_arrivals = df_arrivals.sort_values("Arrival").reset_index(drop=True)
    arrival_times = df_arrivals.set_index("Job")["Arrival"].to_dict()

    # Jobnamen nach Ankunftszeit sortieren (absteigend)
    job_names = list(df_arrivals.sort_values("Arrival", ascending=False)["Job"])

    num_jobs = len(job_names)
    
    all_ops = [job_dict[job_name] for job_name in job_names]


    # Maschinen extrahieren
    all_machines = {op[0] for job in all_ops for op in job}

    # LP-Problem definieren
    prob = pulp.LpProblem("JobShop_Optimal_HiGHS", pulp.LpMinimize)

    # Variablen: Startzeiten
    starts = {
        (j, o): pulp.LpVariable(f"start_{j}_{o}", lowBound=0, cat="Continuous")
        for j in range(num_jobs) for o in range(len(all_ops[j]))
    }

    # Makespan-Variable
    makespan = pulp.LpVariable("makespan", lowBound=0, cat="Continuous")
    prob += makespan  # Ziel 1: Makespan minimieren

    # Technologische Reihenfolge und Ankunftszeit
    for j, job_name in enumerate(job_names):
        job = job_dict[job_name]
        prob += starts[(j, 0)] >= arrival_times[job_name]
        for o in range(1, len(job)):
            d_prev = job[o - 1][1]
            prob += starts[(j, o)] >= starts[(j, o - 1)] + d_prev

    # Maschinenkonflikte (mit kleinem Abstand epsilon)
    bigM = 1e5
    for m in all_machines:
        ops = [(j, o, d) for j in range(num_jobs)
               for o, (mach, d) in enumerate(all_ops[j]) if mach == m]
        for i in range(len(ops)):
            j1, o1, d1 = ops[i]
            for j2, o2, d2 in ops[i + 1:]:
                if j1 != j2:
                    y = pulp.LpVariable(f"y_{j1}_{o1}_{j2}_{o2}", cat="Binary")
                    prob += starts[(j1, o1)] + d1 + epsilon <= starts[(j2, o2)] + bigM * (1 - y)
                    prob += starts[(j2, o2)] + d2 + epsilon <= starts[(j1, o1)] + bigM * y

    # Makespan-Bedingung für jede Job-Endoperation
    for j in range(num_jobs):
        last_op = len(all_ops[j]) - 1
        prob += makespan >= starts[(j, last_op)] + all_ops[j][last_op][1]

    # Solver starten
    solver = pulp.HiGHS_CMD(msg=True, timeLimit=solver_time_limit)
    prob.solve(solver)

    # Ergebnisse extrahieren
    schedule_data = []
    for (j, o), var in sorted(starts.items()):
        start = var.varValue
        if start is not None:
            machine, duration = all_ops[j][o]
            end = start + duration
            schedule_data.append({
                "Job": job_names[j],
                "Machine": f"M{machine}",
                "Start": round(start, 2),
                "Processing Time": duration,
                "End": round(end, 2)
            })

    df_schedule = pd.DataFrame(schedule_data)

    df_schedule["Arrival"] = df_schedule["Job"].map(arrival_times)
    df_schedule["Flow time"] = df_schedule["End"] - df_schedule["Arrival"]

    # Spaltenreihenfolge anpassen
    df_schedule = df_schedule[["Job", "Arrival", "Machine", "Start", "Processing Time", "Flow time", "End"]]

    makespan_value = round(pulp.value(makespan), 3)

    return df_schedule, makespan_value


# Dispatching Rules -----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------

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