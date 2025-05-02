import pulp
import pandas as pd
# conda install -c conda-forge highs



def solve_jssp_total_individual_makespans(job_dict, df_arrivals, solver_time_limit=300, epsilon=0.00):
    """
    Minimiert die Summe der individuellen Makespans (Durchlaufzeiten) aller Jobs:
    Makespan_j = Endzeit_j - Ankunftszeit_j

    Parameter:
    - job_dict: Dictionary mit Jobdaten (jede Operation als (Maschine, Dauer))
    - df_arrivals: DataFrame mit Spalten "Job-ID" und "Ankunftszeit (Minuten)"
    - solver_time_limit: Max. Zeit in Sekunden für den Solver
    - epsilon: Kleiner Abstand zur Vermeidung von Maschinenkonflikten
    """
    
    job_names = list(job_dict.keys())
    num_jobs = len(job_names)
    all_ops = list(job_dict.values())

    # Maschinenmenge extrahieren
    all_machines = {op[0] for job in all_ops for op in job}

    # LP-Problem definieren
    prob = pulp.LpProblem("JobShop_Minimize_Total_Individual_Makespans", pulp.LpMinimize)

    # Startzeit-Variablen für jede Operation
    starts = {
        (j, o): pulp.LpVariable(f"start_{j}_{o}", lowBound=0, cat="Continuous")
        for j in range(num_jobs) for o in range(len(all_ops[j]))
    }

    # Endzeit-Variable pro Job (letzte Operation)
    job_ends = {
        j: pulp.LpVariable(f"job_end_{j}", lowBound=0, cat="Continuous")
        for j in range(num_jobs)
    }

    # Ankunftszeiten als Dictionary
    arrival_times = df_arrivals.set_index("Job-ID")["Ankunftszeit (Minuten)"].to_dict()

    # Ziel: Summe der Durchlaufzeiten (Ende - Ankunft) minimieren
    prob += pulp.lpSum([
        job_ends[j] - arrival_times[job_names[j]]
        for j in range(num_jobs)
    ])

    # Technologische Reihenfolge + Ankunftszeit
    for j, job_name in enumerate(job_names):
        job = job_dict[job_name]
        prob += starts[(j, 0)] >= arrival_times[job_name]
        for o in range(1, len(job)):
            d_prev = job[o - 1][1]
            prob += starts[(j, o)] >= starts[(j, o - 1)] + d_prev

    # Maschinenkonflikte (Disjunktivität mit big-M)
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

    # Endzeit pro Job definieren (letzte Operation)
    for j in range(num_jobs):
        last_op = len(all_ops[j]) - 1
        prob += job_ends[j] >= starts[(j, last_op)] + all_ops[j][last_op][1]

    # Solver ausführen
    solver = pulp.HiGHS_CMD(msg=True, timeLimit=solver_time_limit)
    prob.solve(solver)

    # Ergebnis extrahieren
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
                "Duration": duration,
                "End": round(end, 2)
            })

    df_schedule = pd.DataFrame(schedule_data)

    # Durchlaufzeiten berechnen
    df_schedule["Ankunft"] = df_schedule["Job"].map(arrival_times)
    df_schedule["Durchlaufzeit"] = df_schedule["End"] - df_schedule["Ankunft"]

    # Gesamtdurchlaufzeit (Zielfunktion)
    total_makespan = round(pulp.value(prob.objective), 3)

    return df_schedule, total_makespan


# HiGHs V0 
def solve_jssp_global_makespan(job_dict, df_arrivals, solver_time_limit=300, epsilon=0.00):
    """
    Erste Stufe: Minimierung des Makespan (Gesamtdauer) eines Job-Shop-Problems.

    Parameter:
    - epsilon: Kleiner Sicherheitsabstand (in Minuten) zwischen Operationen auf derselben Maschine,
               um numerische Ungenauigkeiten und Maschinenkonflikte zu vermeiden (z.B. 0.06 Minuten = 3.6 Sekunden).
    """

    job_names = list(job_dict.keys())
    num_jobs = len(job_names)
    all_ops = list(job_dict.values())

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
    arrival_times = df_arrivals.set_index("Job-ID")["Ankunftszeit (Minuten)"].to_dict()

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
                "Duration": duration,
                "End": round(end, 2)
            })

    df_schedule = pd.DataFrame(schedule_data)
    makespan_value = round(pulp.value(makespan), 3)

    return df_schedule, makespan_value