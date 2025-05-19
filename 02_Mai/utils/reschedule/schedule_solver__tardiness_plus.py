# Tardiness & Lateness
import re
import pulp
import pandas as pd

# Tardiness ---------------------------------------------------------------------------------------
# SUM
def solve_jssp_sum_tardiness_with_fixed_ops(
    df_jssp: pd.DataFrame,
    df_arrivals_deadlines: pd.DataFrame,
    df_executed: pd.DataFrame,
    solver_time_limit: int = 1200,
    epsilon: float = 0.0,
    reschedule_start: float = 1440.0
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
    bigM = 1e6
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
    prob.solve(pulp.HiGHS_CMD(msg=True, timeLimit=solver_time_limit))

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
