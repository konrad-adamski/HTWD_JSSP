import pulp
import pandas as pd
# conda install -c conda-forge highs


def solve_jssp_sum_tardiness(job_dict, df_arrivals_deadlines, solver_time_limit=1200, epsilon=0.00):
    """
    Minimiert die Summe der Tardiness (Verspätungen) aller Jobs.
    Zielfunktion: sum_j [ max(0, Endzeit_j - Deadline_j) ]

    Parameter:
    - job_dict: Dictionary mit Jobdaten (jede Operation als (Maschine, Dauer))
    - df_arrivals_deadlines: DataFrame mit Spalten "Job", "Arrival", "Deadline"
    - solver_time_limit: Max. Zeit in Sekunden für den Solver
    - epsilon: Kleiner Abstand zur Vermeidung von Maschinenkonflikten
    """


    # Daten vorbereiten
    df_arrivals_deadlines = df_arrivals_deadlines.sort_values("Arrival").reset_index(drop=True)
    arrival_times = df_arrivals_deadlines.set_index("Job")["Arrival"].to_dict()
    deadlines = df_arrivals_deadlines.set_index("Job")["Deadline"].to_dict()

    job_names = list(df_arrivals_deadlines.sort_values("Arrival", ascending=False)["Job"])

    num_jobs = len(job_names)
    all_ops = [job_dict[job] for job in job_names]
    all_machines = {op[0] for job in all_ops for op in job}

    # LP-Problem definieren
    prob = pulp.LpProblem("JobShop_Total_Tardiness", pulp.LpMinimize)

    # Variablen
    starts = {
        (j, o): pulp.LpVariable(f"start_{j}_{o}", lowBound=0, cat="Continuous")
        for j in range(num_jobs) for o in range(len(all_ops[j]))
    }

    job_ends = {
        j: pulp.LpVariable(f"job_end_{j}", lowBound=0, cat="Continuous")
        for j in range(num_jobs)
    }

    tardiness = {
        j: pulp.LpVariable(f"tardiness_{j}", lowBound=0, cat="Continuous")
        for j in range(num_jobs)
    }

    # Zielfunktion: Summe der Tardiness
    prob += pulp.lpSum([tardiness[j] for j in range(num_jobs)])

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

    # Job-Endzeiten und Tardiness
    for j, job_name in enumerate(job_names):
        last_op = len(all_ops[j]) - 1
    
        # Endzeit = Ende der letzten Operation (genau, nicht nur >=)
        prob += job_ends[j] == starts[(j, last_op)] + all_ops[j][last_op][1]  #prob += job_ends[j] >= starts[(j, last_op)] + all_ops[j][last_op][1]
    
        # Tardiness-Definition
        prob += tardiness[j] >= job_ends[j] - deadlines[job_name]


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
    df_schedule["Deadline"] = df_schedule["Job"].map(deadlines)
    df_schedule["Tardiness"] = df_schedule.groupby("Job")["End"].transform("max") - df_schedule["Deadline"]
    df_schedule["Tardiness"] = df_schedule["Tardiness"].clip(lower=0)

    df_schedule = df_schedule[["Job", "Arrival", "Deadline", "Machine", "Start", "Processing Time", "End", "Tardiness"]]

    total_tardiness = round(pulp.value(prob.objective), 3)
    print(f"Total Tardiness: {total_tardiness}")

    return df_schedule