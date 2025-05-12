import pandas as pd


# minimiert die gewichtete Flow-Time
def solve_jssp_weighted_with_fixed_operations_fast(job_dict, df_arrivals, df_executed,
                                                    solver_time_limit=300, epsilon=0.0,
                                                    arrival_column="Ankunftszeit (Minuten)",
                                                    reschedule_start=1440):
    """
    Schnelle Rescheduling-Variante mit fixierten Operationen.
    Plant alle verbleibenden Jobs ab reschedule_start, unter Berücksichtigung der letzten
    Endzeit bereits ausgeführter Operationen pro Job.
    """

    import pulp
    import pandas as pd

    print("🔁 Starte schnelles Rescheduling ab t =", reschedule_start)

    # Vorbereitung: Ankunftszeiten & Sortierung
    df_arrivals = df_arrivals.sort_values(arrival_column).reset_index(drop=True)
    arrival_times = df_arrivals.set_index("Job")[arrival_column].to_dict()
    job_names = list(df_arrivals.sort_values(arrival_column, ascending=False)["Job"])
    num_jobs = len(job_names)
    all_ops = [job_dict[job] for job in job_names]
    all_machines = {op[0] for job in all_ops for op in job}

    print(f"🔹 {num_jobs} Jobs erkannt")

    # Schnellzugriff vorbereiten
    job_index_map = {name: j for j, name in enumerate(job_names)}
    arrival_lookup = arrival_times

    # Letzte Endzeit bereits geplanter Operationen pro Job
    last_executed_end = df_executed.groupby("Job")["End"].max().to_dict()

    # Fixierte Maschinen-ID extrahieren
    df_executed = df_executed[df_executed["End"] >= reschedule_start].copy()
    df_executed["MachineID"] = df_executed["Machine"].str.extract(r"M(\d+)").astype(int)

    # Fixe Operationen gruppiert pro Maschine
    fixed_ops_dict = {
        m: list(zip(gr["Start"], gr["End"], gr["Job"]))
        for m, gr in df_executed.groupby("MachineID")
    }

    # LP-Modell
    prob = pulp.LpProblem("FastJSSP", pulp.LpMinimize)

    # Startzeit-Variablen
    starts = {
        (j, o): pulp.LpVariable(f"start_{j}_{o}", lowBound=0, cat="Continuous")
        for j in range(num_jobs)
        for o in range(len(all_ops[j]))
    }

    # Endzeit-Variablen je Job
    job_ends = {
        j: pulp.LpVariable(f"job_end_{j}", lowBound=0, cat="Continuous")
        for j in range(num_jobs)
    }

    # Gewichte
    weights = {j: 1 / (1 + arrival_lookup[job_names[j]]) for j in range(num_jobs)}

    # Zielfunktion
    prob += pulp.lpSum([
        weights[j] * (job_ends[j] - arrival_lookup[job_names[j]])
        for j in range(num_jobs)
    ])

    # Technologische Reihenfolge + individuelle Startbedingungen
    for j, job_name in enumerate(job_names):
        job = job_dict[job_name]
        earliest_start = max(
            arrival_lookup[job_name],
            last_executed_end.get(job_name, reschedule_start)
        )
        prob += starts[(j, 0)] >= earliest_start
        for o in range(1, len(job)):
            prob += starts[(j, o)] >= starts[(j, o - 1)] + job[o - 1][1]

    # Maschinenkonflikte (optimiert)
    bigM = 1e5
    for m in sorted(all_machines):
        ops = [(j, o, d) for j in range(num_jobs)
               for o, (mach, d) in enumerate(all_ops[j]) if mach == m]

        # Konflikte zwischen neuen Jobs (nur wenn j1 ≠ j2)
        for i in range(len(ops)):
            j1, o1, d1 = ops[i]
            for j2, o2, d2 in ops[i + 1:]:
                if j1 == j2:
                    continue  # gleicher Job → technologisch geregelt
                y = pulp.LpVariable(f"y_{j1}_{o1}_{j2}_{o2}", cat="Binary")
                prob += starts[(j1, o1)] + d1 + epsilon <= starts[(j2, o2)] + bigM * (1 - y)
                prob += starts[(j2, o2)] + d2 + epsilon <= starts[(j1, o1)] + bigM * y

        # Konflikte mit fixierten Operationen
        for j1, o1, d1 in ops:
            job1 = job_names[j1]
            job1_arrival = arrival_lookup.get(job1, 0)

            for fixed_start, fixed_end, fixed_job in fixed_ops_dict.get(m, []):
                if fixed_end + epsilon < job1_arrival:
                    continue
                y_fix = pulp.LpVariable(f"y_fix_{j1}_{o1}_{fixed_job}", cat="Binary")
                prob += starts[(j1, o1)] + d1 + epsilon <= fixed_start + bigM * (1 - y_fix)
                prob += fixed_end + epsilon <= starts[(j1, o1)] + bigM * y_fix

    # Endzeitbindung je Job
    for j in range(num_jobs):
        last_op = len(all_ops[j]) - 1
        prob += job_ends[j] >= starts[(j, last_op)] + all_ops[j][last_op][1]

    # Solver
    solver = pulp.HiGHS_CMD(msg=True, timeLimit=solver_time_limit)
    prob.solve(solver)

    # Ergebnis
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
    df_schedule["Arrival"] = df_schedule["Job"].map(arrival_lookup)
    df_schedule["Flow time"] = df_schedule["End"] - df_schedule["Arrival"]
    df_schedule = df_schedule[["Job", "Arrival", "Machine", "Start", "Processing Time", "Flow time", "End"]]

    #total_weighted_flowtime = round(pulp.value(prob.objective), 3)
    print("✅ Fertig!")
    return df_schedule