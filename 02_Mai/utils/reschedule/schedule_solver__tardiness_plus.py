# Tardiness & Lateness
import re
import pulp
import pandas as pd

import utils.reschedule.bigM_estimator as bigM_func

def solve_jssp_bi_criteria_sum_tardiness_deviation_with_fixed_ops(
    df_jssp: pd.DataFrame,
    df_arrivals_deadlines: pd.DataFrame,
    df_executed: pd.DataFrame,
    df_original_plan: pd.DataFrame,
    r: float = 0.5,
    solver_time_limit: int = 300,
    epsilon: float = 0.0,
    arrival_column: str = "Arrival",
    deadline_column: str = "Deadline",
    reschedule_start: float = 1440.0, 
    msg_print=False, threads=None
) -> pd.DataFrame:
    """
    Bi-kriterielles Rescheduling: Tardiness + Planabweichung mit fixierten Operationen.
    Zielfunktion: Z(σ) = r * sum(Tardiness) + (1 - r) * sum(Deviation)

    Rückgabe:
    - df_schedule mit ['Job','Operation','Arrival','Deadline','Machine','Start',
                      'Processing Time','End','Tardiness']
    """
    df_arrivals_deadlines = df_arrivals_deadlines.sort_values(arrival_column).reset_index(drop=True)
    arrival = df_arrivals_deadlines.set_index("Job")[arrival_column].to_dict()
    deadline = df_arrivals_deadlines.set_index("Job")[deadline_column].to_dict()
    job_list = df_arrivals_deadlines["Job"].tolist()
    num_jobs = len(job_list)

    original_start_times = {
        (row["Job"], row["Operation"]): row["Start"]
        for _, row in df_original_plan.iterrows()
    }

    ops_grouped = df_jssp.sort_values(["Job", "Operation"]).groupby("Job")
    all_ops, all_machines = [], set()
    for job in job_list:
        grp = ops_grouped.get_group(job)
        ops = []
        for _, row in grp.iterrows():
            op_id = row["Operation"]
            m_id = int(re.search(r"M(\d+)", str(row["Machine"])).group(1))
            dur = float(row["Processing Time"])
            ops.append((op_id, m_id, dur))
            all_machines.add(m_id)
        all_ops.append(ops)

    df_executed_fixed = df_executed[df_executed['End'] >= reschedule_start].copy()
    df_executed_fixed['MachineID'] = df_executed_fixed['Machine'].astype(str).str.extract(r"M(\d+)", expand=False).astype(int)
    fixed_ops = {
        m: list(gr[['Start', 'End', 'Job']].itertuples(index=False, name=None))
        for m, gr in df_executed_fixed.groupby('MachineID')
    }
    last_executed_end = df_executed.groupby('Job')['End'].max().to_dict()

    prob = pulp.LpProblem("JSSP_BiCriteria_Tardiness_Deviation_Fixed", pulp.LpMinimize)

    starts = {
        (j, o): pulp.LpVariable(f"start_{j}_{o}", lowBound=0)
        for j in range(num_jobs)
        for o in range(len(all_ops[j]))
    }
    job_ends = {
        j: pulp.LpVariable(f"end_{j}", lowBound=0)
        for j in range(num_jobs)
    }
    tardiness = {
        j: pulp.LpVariable(f"tard_{j}", lowBound=0)
        for j in range(num_jobs)
    }

    # Tardiness-Teil
    for j, job in enumerate(job_list):
        last_o = len(all_ops[j]) - 1
        dur = all_ops[j][last_o][2]
        prob += job_ends[j] == starts[(j, last_o)] + dur
        prob += tardiness[j] >= job_ends[j] - deadline[job]

    sum_tardiness = pulp.lpSum(tardiness[j] for j in range(num_jobs))

    # Deviation-Teil
    deviations = {}
    for (j, o) in starts:
        job = job_list[j]
        key = (job, o)
        if key in original_start_times:
            dev = pulp.LpVariable(f"dev_{j}_{o}", lowBound=0)
            prob += dev >= starts[(j, o)] - original_start_times[key]
            prob += dev >= original_start_times[key] - starts[(j, o)]
            deviations[(j, o)] = dev
    sum_deviation = pulp.lpSum(deviations.values())

    # Zielfunktion
    prob += r * sum_tardiness + (1 - r) * sum_deviation

    # Reihenfolgen und frühester Start
    for j, job in enumerate(job_list):
        earliest = max(arrival[job], last_executed_end.get(job, reschedule_start))
        prob += starts[(j, 0)] >= earliest
        for o in range(1, len(all_ops[j])):
            _, _, dur_prev = all_ops[j][o-1]
            prob += starts[(j, o)] >= starts[(j, o-1)] + dur_prev

    # Maschinenkonflikte
    M = bigM_func.estimate_bigM_with_deadline_and_original_plan(df_jssp, df_arrivals_deadlines, df_original_plan)
    
    for m in sorted(all_machines):
        ops_on_m = [
            (j, o, all_ops[j][o][2])
            for j in range(num_jobs)
            for o, (_, mach, _) in enumerate(all_ops[j])
            if mach == m
        ]
        for i in range(len(ops_on_m)):
            j1, o1, d1 = ops_on_m[i]
            for j2, o2, d2 in ops_on_m[i+1:]:
                if j1 == j2:
                    continue
                y = pulp.LpVariable(f"y_{j1}_{o1}_{j2}_{o2}", cat="Binary")
                prob += starts[(j1, o1)] + d1 + epsilon <= starts[(j2, o2)] + M*(1 - y)
                prob += starts[(j2, o2)] + d2 + epsilon <= starts[(j1, o1)] + M*y
        for j1, o1, d1 in ops_on_m:
            for fixed_start, fixed_end, _ in fixed_ops.get(m, []):
                y_fix = pulp.LpVariable(f"y_fix_{j1}_{o1}_{fixed_start}", cat='Binary')
                prob += starts[(j1, o1)] + d1 + epsilon <= fixed_start + M*(1 - y_fix)
                prob += fixed_end + epsilon <= starts[(j1, o1)] + M*y_fix

    # Lösen
    if threads:
        prob.solve(pulp.HiGHS_CMD(msg=msg_print, timeLimit=solver_time_limit, threads=threads))
    else:
        prob.solve(pulp.HiGHS_CMD(msg=msg_print, timeLimit=solver_time_limit))

    # Ergebnisse
    recs = []
    for (j, o), var in sorted(starts.items()):
        st = var.varValue
        if st is None:
            continue
        op_id, mach, dur = all_ops[j][o]
        end = st + dur
        job = job_list[j]
        recs.append({
            "Job": job,
            "Operation": op_id,
            "Arrival": arrival[job],
            "Deadline": deadline[job],
            "Machine": f"M{mach}",
            "Start": round(st, 2),
            "Processing Time": dur,
            "End": round(end, 2),
            "Tardiness": max(0.0, round(end - deadline[job], 2))
        })

    df_schedule = pd.DataFrame(recs)
    df_schedule = df_schedule[['Job', 'Operation', 'Arrival', 'Deadline', 'Machine',
                               'Start', 'Processing Time', 'End', 'Tardiness']]
    df_schedule = df_schedule.sort_values(['Arrival', 'Start']).reset_index(drop=True)

    # Log
    print("\nSolver-Informationen:")
    print(f"  Zielfunktionswert       : {round(pulp.value(prob.objective), 4)}")
    print(f"  Solver-Status           : {pulp.LpStatus[prob.status]}")
    print(f"  Anzahl Variablen        : {len(prob.variables())}")
    print(f"  Anzahl Constraints      : {len(prob.constraints)}")

    return df_schedule



# Tardiness ---------------------------------------------------------------------------------------
# SUM
def solve_jssp_sum_tardiness_with_fixed_ops(
    df_jssp: pd.DataFrame,
    df_arrivals_deadlines: pd.DataFrame,
    df_executed: pd.DataFrame,
    solver_time_limit: int = 1200,
    epsilon: float = 0.0,
    reschedule_start: float = 1440.0,
    msg_print=False, threads=None
) -> pd.DataFrame:
    """
    Minimiert die Summe der Tardiness unter Berücksichtigung bereits ausgeführter Operationen (fixe Belegung).

    Parameter:
    - df_jssp: DataFrame mit ['Job','Operation','Machine','Processing Time']
    - df_arrivals_deadlines: DataFrame mit ['Job','Arrival','Deadline']
    - df_executed: DataFrame mit ['Job','Machine','Start','End']
    - solver_time_limit: Zeitlimit für Solver
    - epsilon: Puffer zwischen Maschinenoperationen
    - reschedule_start: Zeitpunkt, ab dem neu geplant wird

    Rückgabe:
    - df_schedule: DataFrame mit ['Job','Operation','Arrival','Deadline','Machine',
                                  'Start','Processing Time','End','Tardiness']
    """
    df_arr = df_arrivals_deadlines.sort_values("Arrival").reset_index(drop=True)
    arrival = df_arr.set_index("Job")["Arrival"].to_dict()
    deadline = df_arr.set_index("Job")["Deadline"].to_dict()
    jobs = df_arr["Job"].tolist()
    n = len(jobs)

    # Job-Operationen sammeln
    ops_grouped = df_jssp.sort_values(["Job","Operation"]).groupby("Job")
    all_ops, machines = [], set()
    for job in jobs:
        seq = []
        for _, row in ops_grouped.get_group(job).iterrows():
            op_id = row["Operation"]
            m = int(re.search(r"M(\d+)", str(row["Machine"])).group(1))
            d = float(row["Processing Time"])
            seq.append((op_id, m, d))
            machines.add(m)
        all_ops.append(seq)

    # Bereits fixierte Operationen
    last_executed_end = df_executed.groupby("Job")["End"].max().to_dict()
    df_executed_fixed = df_executed[df_executed["End"] >= reschedule_start].copy()
    df_executed_fixed["MachineID"] = (
        df_executed_fixed["Machine"].astype(str)
        .str.extract(r"M(\d+)", expand=False)
        .astype(int)
    )
    fixed_ops = {
        m: list(gr[["Start", "End", "Job"]].itertuples(index=False, name=None))
        for m, gr in df_executed_fixed.groupby("MachineID")
    }

    # Modell
    prob = pulp.LpProblem("JSSP_Sum_Tardiness_FixedOps", pulp.LpMinimize)
    starts = {
        (j, o): pulp.LpVariable(f"start_{j}_{o}", lowBound=0)
        for j in range(n) for o in range(len(all_ops[j]))
    }
    ends = {
        j: pulp.LpVariable(f"end_{j}", lowBound=0)
        for j in range(n)
    }
    tard = {
        j: pulp.LpVariable(f"tardiness_{j}", lowBound=0)
        for j in range(n)
    }

    # Zielfunktion
    prob += pulp.lpSum(tard[j] for j in range(n))

    # Technologische Reihenfolge & Startbedingungen
    for j, job in enumerate(jobs):
        earliest = max(arrival[job], last_executed_end.get(job, reschedule_start))
        prob += starts[(j, 0)] >= earliest
        for o in range(1, len(all_ops[j])):
            _, _, d_prev = all_ops[j][o - 1]
            prob += starts[(j, o)] >= starts[(j, o - 1)] + d_prev
        # Endzeit und Tardiness
        _, _, d_last = all_ops[j][-1]
        prob += ends[j] == starts[(j, len(all_ops[j]) - 1)] + d_last
        prob += tard[j] >= ends[j] - deadline[job]

    # Maschinenkonflikte
    bigM = bigM_func.estimate_bigM_with_deadline(df_jssp, df_arrivals_deadlines)
    for m in machines:
        ops_on_m = [
            (j, o, all_ops[j][o][2])
            for j in range(n)
            for o in range(len(all_ops[j]))
            if all_ops[j][o][1] == m
        ]
        for i in range(len(ops_on_m)):
            j1, o1, d1 = ops_on_m[i]
            for j2, o2, d2 in ops_on_m[i+1:]:
                if j1 == j2:
                    continue
                y = pulp.LpVariable(f"y_{j1}_{o1}_{j2}_{o2}", cat="Binary")
                prob += starts[(j1, o1)] + d1 + epsilon <= starts[(j2, o2)] + bigM * (1 - y)
                prob += starts[(j2, o2)] + d2 + epsilon <= starts[(j1, o1)] + bigM * y
            # Fixierte Ops
            for fixed_start, fixed_end, fixed_job in fixed_ops.get(m, []):
                y_fix = pulp.LpVariable(f"y_fix_{j1}_{o1}_{fixed_job}", cat="Binary")
                prob += starts[(j1, o1)] + d1 + epsilon <= fixed_start + bigM * (1 - y_fix)
                prob += fixed_end + epsilon <= starts[(j1, o1)] + bigM * y_fix

    # Lösen
    if threads:
        prob.solve(pulp.HiGHS_CMD(msg=msg_print, timeLimit=solver_time_limit, threads=threads))
    else:
        prob.solve(pulp.HiGHS_CMD(msg=msg_print, timeLimit=solver_time_limit))

    # Ergebnis extrahieren
    records = []
    for j, job in enumerate(jobs):
        for o, (op_id, m, d) in enumerate(all_ops[j]):
            st = starts[(j, o)].varValue
            if st is None:
                continue
            ed = st + d
            records.append({
                "Job": job,
                "Operation": op_id,
                "Arrival": arrival[job],
                "Deadline": deadline[job],
                "Machine": f"M{m}",
                "Start": round(st, 2),
                "Processing Time": d,
                "End": round(ed, 2),
                "Tardiness": max(0, round(ed - deadline[job], 2))
            })

    df_schedule = pd.DataFrame(records).sort_values(["Arrival", "Start", "Job"]).reset_index(drop=True)

    # Log
    print("\nSolver-Informationen:")
    print(f"  Zielfunktionswert       : {round(pulp.value(prob.objective), 4)}")
    print(f"  Solver-Status           : {pulp.LpStatus[prob.status]}")
    print(f"  Anzahl Variablen        : {len(prob.variables())}")
    print(f"  Anzahl Constraints      : {len(prob.constraints)}")
    
    return df_schedule



# MAX

def solve_jssp_max_tardiness_with_fixed_ops(
    df_jssp: pd.DataFrame,
    df_arrivals_deadlines: pd.DataFrame,
    df_executed: pd.DataFrame,
    solver_time_limit: int = 1200,
    epsilon: float = 0.0,
    reschedule_start: float = 1440.0
) -> pd.DataFrame:
    """
    Minimiert die maximale Tardiness (Verspätung) unter Berücksichtigung fixierter Operationen.

    Rückgabe:
    - df_schedule: DataFrame mit ['Job','Operation','Arrival','Deadline','Machine',
      'Start','Processing Time','End','Tardiness']
    """
    df_arr = df_arrivals_deadlines.sort_values("Arrival").reset_index(drop=True)
    arrival = df_arr.set_index("Job")["Arrival"].to_dict()
    deadline = df_arr.set_index("Job")["Deadline"].to_dict()
    jobs = df_arr["Job"].tolist()

    ops_grouped = df_jssp.sort_values(["Job","Operation"]).groupby("Job")
    all_ops, machines = [], set()
    for job in jobs:
        seq = []
        for _, row in ops_grouped.get_group(job).iterrows():
            op_id = row["Operation"]
            m = int(re.search(r"M(\d+)", str(row["Machine"])).group(1))
            d = float(row["Processing Time"])
            seq.append((op_id, m, d))
            machines.add(m)
        all_ops.append(seq)

    last_executed_end = df_executed.groupby("Job")["End"].max().to_dict()
    df_executed_fixed = df_executed[df_executed["End"] >= reschedule_start].copy()
    df_executed_fixed["MachineID"] = (
        df_executed_fixed["Machine"].astype(str)
        .str.extract(r"M(\d+)", expand=False)
        .astype(int)
    )
    fixed_ops = {
        m: list(gr[["Start", "End", "Job"]].itertuples(index=False, name=None))
        for m, gr in df_executed_fixed.groupby("MachineID")
    }

    n = len(jobs)
    bigM = 1e6

    prob = pulp.LpProblem("JSSP_Max_Tardiness_Fixed", pulp.LpMinimize)

    starts = {
        (j, o): pulp.LpVariable(f"start_{j}_{o}", lowBound=0)
        for j in range(n) for o in range(len(all_ops[j]))
    }
    ends = {j: pulp.LpVariable(f"end_{j}", lowBound=0) for j in range(n)}
    tard = {j: pulp.LpVariable(f"tardiness_{j}", lowBound=0) for j in range(n)}
    max_tardiness = pulp.LpVariable("max_tardiness", lowBound=0)

    prob += max_tardiness

    for j, job in enumerate(jobs):
        seq = all_ops[j]
        earliest = max(arrival[job], last_executed_end.get(job, reschedule_start))
        prob += starts[(j, 0)] >= earliest
        for o in range(1, len(seq)):
            _, _, d_prev = seq[o - 1]
            prob += starts[(j, o)] >= starts[(j, o - 1)] + d_prev
        _, _, d_last = seq[-1]
        prob += ends[j] == starts[(j, len(seq)-1)] + d_last
        prob += tard[j] >= ends[j] - deadline[job]
        prob += max_tardiness >= tard[j]

    for m in machines:
        ops_on_m = [
            (j, o, all_ops[j][o][2])
            for j in range(n)
            for o in range(len(all_ops[j]))
            if all_ops[j][o][1] == m
        ]
        for i in range(len(ops_on_m)):
            j1, o1, d1 = ops_on_m[i]
            for j2, o2, d2 in ops_on_m[i+1:]:
                if j1 == j2:
                    continue
                y = pulp.LpVariable(f"y_{j1}_{o1}_{j2}_{o2}", cat="Binary")
                prob += starts[(j1, o1)] + d1 + epsilon <= starts[(j2, o2)] + bigM*(1-y)
                prob += starts[(j2, o2)] + d2 + epsilon <= starts[(j1, o1)] + bigM*y
            for fixed_start, fixed_end, fixed_job in fixed_ops.get(m, []):
                y_fix = pulp.LpVariable(f"y_fix_{j1}_{o1}_{fixed_job}", cat="Binary")
                prob += starts[(j1, o1)] + d1 + epsilon <= fixed_start + bigM*(1 - y_fix)
                prob += fixed_end + epsilon <= starts[(j1, o1)] + bigM*y_fix

    prob.solve(pulp.HiGHS_CMD(msg=True, timeLimit=solver_time_limit))

    records = []
    for j, job in enumerate(jobs):
        for o, (op_id, m, d) in enumerate(all_ops[j]):
            st = starts[(j, o)].varValue
            if st is None:
                continue
            ed = st + d
            records.append({
                "Job": job,
                "Operation": op_id,
                "Arrival": arrival[job],
                "Deadline": deadline[job],
                "Machine": f"M{m}",
                "Start": round(st, 2),
                "Processing Time": d,
                "End": round(ed, 2),
                "Tardiness": max(0, round(ed - deadline[job], 2))
            })

    df_schedule = pd.DataFrame.from_records(records).sort_values(["Arrival", "Start", "Job"]).reset_index(drop=True)
    return df_schedule



# Lateness ---------------------------------------------------------------------------------------
def solve_jssp_max_absolute_lateness_with_fixed_ops(
    df_jssp: pd.DataFrame,
    df_arrivals_deadlines: pd.DataFrame,
    df_executed: pd.DataFrame,
    solver_time_limit: int = 1200,
    epsilon: float = 0.0,
    reschedule_start: float = 1440.0
) -> pd.DataFrame:
    """
    Minimiert die maximale absolute Lateness unter Berücksichtigung fixierter Operationen.

    Rückgabe:
    - df_schedule: DataFrame mit ['Job','Operation','Arrival','Deadline','Machine',
      'Start','Processing Time','End','Lateness','Absolute Lateness']
    """
    df_arr = df_arrivals_deadlines.sort_values("Arrival").reset_index(drop=True)
    arrival = df_arr.set_index("Job")["Arrival"].to_dict()
    deadline = df_arr.set_index("Job")["Deadline"].to_dict()
    jobs = df_arr["Job"].tolist()

    ops_grouped = df_jssp.sort_values(["Job","Operation"]).groupby("Job")
    all_ops, machines = [], set()
    for job in jobs:
        seq = []
        for _, row in ops_grouped.get_group(job).iterrows():
            op_id = row["Operation"]
            m = int(re.search(r"M(\d+)", str(row["Machine"])).group(1))
            d = float(row["Processing Time"])
            seq.append((op_id, m, d))
            machines.add(m)
        all_ops.append(seq)

    last_executed_end = df_executed.groupby("Job")["End"].max().to_dict()
    df_executed_fixed = df_executed[df_executed["End"] >= reschedule_start].copy()
    df_executed_fixed["MachineID"] = (
        df_executed_fixed["Machine"].astype(str)
        .str.extract(r"M(\d+)", expand=False)
        .astype(int)
    )
    fixed_ops = {
        m: list(gr[["Start", "End", "Job"]].itertuples(index=False, name=None))
        for m, gr in df_executed_fixed.groupby("MachineID")
    }

    n = len(jobs)
    bigM = 1e6

    prob = pulp.LpProblem("JSSP_Max_Abs_Lateness_Fixed", pulp.LpMinimize)

    starts = {
        (j, o): pulp.LpVariable(f"start_{j}_{o}", lowBound=0)
        for j in range(n) for o in range(len(all_ops[j]))
    }
    ends = {j: pulp.LpVariable(f"end_{j}", lowBound=0) for j in range(n)}
    abs_lateness = {j: pulp.LpVariable(f"abs_lateness_{j}", lowBound=0) for j in range(n)}
    max_abs_lateness = pulp.LpVariable("max_abs_lateness", lowBound=0)

    prob += max_abs_lateness

    for j, job in enumerate(jobs):
        seq = all_ops[j]
        earliest = max(arrival[job], last_executed_end.get(job, reschedule_start))
        prob += starts[(j, 0)] >= earliest
        for o in range(1, len(seq)):
            _, _, d_prev = seq[o - 1]
            prob += starts[(j, o)] >= starts[(j, o - 1)] + d_prev
        _, _, d_last = seq[-1]
        prob += ends[j] == starts[(j, len(seq)-1)] + d_last
        lateness = ends[j] - deadline[job]
        prob += abs_lateness[j] >= lateness
        prob += abs_lateness[j] >= -lateness
        prob += max_abs_lateness >= abs_lateness[j]

    for m in machines:
        ops_on_m = [
            (j, o, all_ops[j][o][2])
            for j in range(n)
            for o in range(len(all_ops[j]))
            if all_ops[j][o][1] == m
        ]
        for i in range(len(ops_on_m)):
            j1, o1, d1 = ops_on_m[i]
            for j2, o2, d2 in ops_on_m[i+1:]:
                if j1 == j2:
                    continue
                y = pulp.LpVariable(f"y_{j1}_{o1}_{j2}_{o2}", cat="Binary")
                prob += starts[(j1, o1)] + d1 + epsilon <= starts[(j2, o2)] + bigM*(1 - y)
                prob += starts[(j2, o2)] + d2 + epsilon <= starts[(j1, o1)] + bigM*y
            for fixed_start, fixed_end, fixed_job in fixed_ops.get(m, []):
                y_fix = pulp.LpVariable(f"y_fix_{j1}_{o1}_{fixed_job}", cat="Binary")
                prob += starts[(j1, o1)] + d1 + epsilon <= fixed_start + bigM*(1 - y_fix)
                prob += fixed_end + epsilon <= starts[(j1, o1)] + bigM*y_fix

    prob.solve(pulp.HiGHS_CMD(msg=True, timeLimit=solver_time_limit))

    records = []
    for j, job in enumerate(jobs):
        for o, (op_id, m, d) in enumerate(all_ops[j]):
            st = starts[(j, o)].varValue
            if st is None:
                continue
            ed = st + d
            late = round(ed - deadline[job], 2)
            records.append({
                "Job": job,
                "Operation": op_id,
                "Arrival": arrival[job],
                "Deadline": deadline[job],
                "Machine": f"M{m}",
                "Start": round(st, 2),
                "Processing Time": d,
                "End": round(ed, 2),
                "Lateness": late,
                "Absolute Lateness": abs(late)
            })

    df_schedule = pd.DataFrame.from_records(records).sort_values(["Arrival", "Start", "Job"]).reset_index(drop=True)
    return df_schedule
