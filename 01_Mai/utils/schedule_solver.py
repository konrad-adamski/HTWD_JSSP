import pulp
import pandas as pd
# conda install -c conda-forge highs

#  ungewichtet
def solve_jssp_individual_flowtime(job_dict, df_arrivals, solver_time_limit=300, epsilon=0.00):
    """
    Minimiert die Summe der individuellen Durchlaufzeiten (Flow Times) aller Jobs.
    Zielfunktion: sum_j [ Endzeit_j - Ankunftszeit_j ]

    Parameter:
    - job_dict: Dictionary mit Jobdaten (jede Operation als (Maschine, Dauer))
    - df_arrivals: DataFrame mit Spalten "Job-ID" und "Ankunftszeit (Minuten)"
    - solver_time_limit: Max. Zeit in Sekunden für den Solver
    - epsilon: Kleiner Abstand zur Vermeidung von Maschinenkonflikten
    """

    # Ankunftszeiten als Dictionary
    df_arrivals = df_arrivals.sort_values("Ankunftszeit (Minuten)").reset_index(drop=True)
    arrival_times = df_arrivals.set_index("Job-ID")["Ankunftszeit (Minuten)"].to_dict()

    # Jobnamen nach Ankunftszeit sortieren (absteigend)
    job_names = list(df_arrivals.sort_values("Ankunftszeit (Minuten)", ascending=False)["Job-ID"])

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

# gewichtet
def solve_jssp_weighted_individual_flowtime(job_dict, df_arrivals, solver_time_limit=300, epsilon=0.00):
    """
    Minimiert die gewichtete Summe der individuellen Durchlaufzeiten (Flow Times) aller Jobs.
    Gewichtung bevorzugt früh ankommende Jobs.

    Gewicht_j = 1 / (1 + Ankunftszeit_j)
    Zielfunktion: sum_j [ Gewicht_j * (Endzeit_j - Ankunftszeit_j) ]

    Parameter:
    - job_dict: Dictionary mit Jobdaten (jede Operation als (Maschine, Dauer))
    - df_arrivals: DataFrame mit Spalten "Job-ID" und "Ankunftszeit (Minuten)"
    - solver_time_limit: Max. Zeit in Sekunden für den Solver
    - epsilon: Kleiner Abstand zur Vermeidung von Maschinenkonflikten
    """

    # Ankunftszeiten als Dictionary
    df_arrivals = df_arrivals.sort_values("Ankunftszeit (Minuten)").reset_index(drop=True)
    arrival_times = df_arrivals.set_index("Job-ID")["Ankunftszeit (Minuten)"].to_dict()

    # Jobnamen nach Ankunftszeit sortieren (absteigend)
    job_names = list(df_arrivals.sort_values("Ankunftszeit (Minuten)", ascending=False)["Job-ID"])

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


def solve_jssp_weighted_individual_flowtime_v2(job_dict, df_arrivals, solver_time_limit=300, epsilon=0.00, arrival_column="Ankunftszeit (Minuten)"):
    """
    Minimiert die gewichtete Summe der individuellen Durchlaufzeiten (Flow Times) aller Jobs.
    Gewichtung bevorzugt früh ankommende Jobs.

    Gewicht_j = 1 / (1 + Ankunftszeit_j)
    Zielfunktion: sum_j [ Gewicht_j * (Endzeit_j - Ankunftszeit_j) ]

    Parameter:
    - job_dict: Dictionary mit Jobdaten (jede Operation als (Maschine, Dauer))
    - df_arrivals: DataFrame mit Spalten "Job-ID" und Ankunftszeit
    - arrival_column: Name der Spalte mit den Ankunftszeiten
    - solver_time_limit: Max. Zeit in Sekunden für den Solver
    - epsilon: Kleiner Abstand zur Vermeidung von Maschinenkonflikten
    """

    # Ankunftszeiten als Dictionary
    df_arrivals = df_arrivals.sort_values(arrival_column).reset_index(drop=True)
    arrival_times = df_arrivals.set_index("Job")[arrival_column].to_dict()

    # Jobnamen nach Ankunftszeit sortieren (absteigend)
    job_names = list(df_arrivals.sort_values(arrival_column, ascending=False)["Job"])

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


## Two Stages --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------


def solve_jssp_global_makespan(job_dict, df_arrivals, solver_time_limit=300, epsilon=0.00):
    """
    Erste Stufe: Minimierung des Makespan (Gesamtdauer) eines Job-Shop-Problems.

    Parameter:
    - epsilon: Kleiner Sicherheitsabstand (in Minuten) zwischen Operationen auf derselben Maschine,
               um numerische Ungenauigkeiten und Maschinenkonflikte zu vermeiden (z.B. 0.06 Minuten = 3.6 Sekunden).
    """


    # Ankunftszeiten als Dictionary
    df_arrivals = df_arrivals.sort_values("Ankunftszeit (Minuten)").reset_index(drop=True)
    arrival_times = df_arrivals.set_index("Job-ID")["Ankunftszeit (Minuten)"].to_dict()

    # Jobnamen nach Ankunftszeit sortieren (absteigend)
    job_names = list(df_arrivals.sort_values("Ankunftszeit (Minuten)", ascending=False)["Job-ID"])

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

    # Ankunftszeiten berücksichtigen
    #arrival_times = df_arrivals.set_index("Job-ID")["Ankunftszeit (Minuten)"].to_dict()

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



def solve_stage2_early_starts_all(job_dict, df_arrivals, optimal_makespan, solver_time_limit=300, epsilon=0.06):
    """
    Zweite Stufe: Minimierung der Summe aller Startzeiten unter Fixierung des Makespan (verbesserte Version).

    Parameter:
    - job_dict: Dictionary der Jobs mit Maschinen- und Bearbeitungszeiten.
    - df_arrivals: DataFrame mit Ankunftszeiten der Jobs.
    - optimal_makespan: Optimales Makespan aus Stufe 1 (fester Endzeitpunkt).
    - solver_time_limit: Zeitlimit für den Solver (Sekunden).
    - epsilon: Kleiner Sicherheitsabstand zwischen Operationen auf derselben Maschine.
    """

    # Ankunftszeiten als Dictionary
    df_arrivals = df_arrivals.sort_values("Ankunftszeit (Minuten)").reset_index(drop=True)
    arrival_times = df_arrivals.set_index("Job-ID")["Ankunftszeit (Minuten)"].to_dict()

    # Jobnamen nach Ankunftszeit sortieren (absteigend)
    job_names = list(df_arrivals.sort_values("Ankunftszeit (Minuten)", ascending=False)["Job-ID"])

    num_jobs = len(job_names)
    
    all_ops = [job_dict[job_name] for job_name in job_names]

    all_machines = {op[0] for job in all_ops for op in job}

    prob = pulp.LpProblem("JobShop_Secondary_EarlyStart", pulp.LpMinimize)

    # Startzeit-Variablen
    starts = {
        (j, o): pulp.LpVariable(f"start_{j}_{o}", lowBound=0, cat="Continuous")
        for j in range(num_jobs) for o in range(len(all_ops[j]))
    }

    # Ankunftszeiten vorbereiten
    #arrival_times = df_arrivals.set_index("Job-ID")["Ankunftszeit (Minuten)"].to_dict()

    # Technologische Reihenfolge und Ankunftszeiten einfügen
    for j, job_name in enumerate(job_names):
        job = job_dict[job_name]
        prob += starts[(j, 0)] >= arrival_times[job_name]
        for o in range(1, len(job)):
            d_prev = job[o - 1][1]
            prob += starts[(j, o)] >= starts[(j, o - 1)] + d_prev

    # Maschinenkonflikte (mit epsilon Abstand)
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

    # Fixierter Makespan: Endzeit der letzten Operationen darf den optimalen Makespan nicht überschreiten
    for j in range(num_jobs):
        last_op = len(all_ops[j]) - 1
        prob += starts[(j, last_op)] + all_ops[j][last_op][1] <= optimal_makespan

    # ZIEL: Minimierung der Summe aller Startzeiten (nicht nur der ersten Operationen)
    total_start = pulp.lpSum([starts[(j, o)] for j in range(num_jobs) for o in range(len(all_ops[j]))])
    prob += total_start

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
    
    return df_schedule, optimal_makespan

