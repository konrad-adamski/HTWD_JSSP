# Tardiness & Lateness
from typing import Tuple
import re
import pulp
import pandas as pd

import math

# Tardiness ---------------------------------------------------------------------------------------
# SUM
import pandas as pd
import pulp
import math
import re

def solve_jssp_sum_tardiness(
    df_jssp: pd.DataFrame,
    df_arrivals_deadlines: pd.DataFrame,
    solver_time_limit: int = 1200,
    epsilon: float = 0.6,
    msg_print: bool = False,
    threads: int = None,
    sort_ascending: bool = False
) -> pd.DataFrame:
    """
    Minimiert die Summe der Tardiness (Verspätungen) aller Jobs.
    Zielfunktion: sum_j [ max(0, Endzeit_j - Deadline_j) ]

    Parameter:
    - df_jssp: DataFrame mit ['Job','Operation','Machine','Processing Time'].
    - df_arrivals_deadlines: DataFrame mit ['Job','Arrival','Deadline'].
    - solver_time_limit: Max. Zeit in Sekunden für den Solver.
    - epsilon: Puffer zwischen Operationen auf derselben Maschine.

    Rückgabe:
    - df_schedule: DataFrame mit ['Job','Operation','Arrival','Deadline',
      'Machine','Start','Processing Time','End','Tardiness'].
    """
    # Vorverarbeitung
    df_arrivals_deadlines = df_arrivals_deadlines.sort_values("Deadline", ascending=sort_ascending).reset_index(drop=True)
    arrival = df_arrivals_deadlines.set_index("Job")["Arrival"].to_dict()
    deadline = df_arrivals_deadlines.set_index("Job")["Deadline"].to_dict()
    jobs = df_arrivals_deadlines["Job"].tolist()

    # Infos für bigM
    max_deadline = max(deadline.values())
    max_proc_time = max(df_jssp["Processing Time"])
    min_arrival = min(arrival.values())

    bigM_raw = max_deadline - min_arrival + max_proc_time*4
    bigM = math.ceil(bigM_raw / 100) * 100 *2
    print(f"BigM: {bigM}")

    # Operationen je Job
    ops_grouped = df_jssp.sort_values(["Job", "Operation"]).groupby("Job")
    all_ops = []
    machines = set()
    for job in jobs:
        seq = []
        for _, row in ops_grouped.get_group(job).iterrows():
            op_id = row["Operation"]
            m = int(re.search(r"M(\d+)", str(row["Machine"])).group(1))
            d = float(row["Processing Time"])
            seq.append((op_id, m, d))
            machines.add(m)
        all_ops.append(seq)

    n = len(jobs)

    # LP-Problem
    prob = pulp.LpProblem("JSSP_Sum_Tardiness", pulp.LpMinimize)

    # Variablen
    starts = {
        (j, o): pulp.LpVariable(
            f"start_{j}_{o}",
            lowBound=arrival[jobs[j]]
        )
        for j in range(n)
        for o in range(len(all_ops[j]))
    }

    ends = {
        j: pulp.LpVariable(
            f"end_{j}",
            lowBound=arrival[jobs[j]]
        )
        for j in range(n)
    }

    tard = {
        j: pulp.LpVariable(f"tardiness_{j}", lowBound=0)
        for j in range(n)
    }

    # Zielfunktion
    prob += pulp.lpSum(tard[j] for j in range(n))

    
    # Technologische Reihenfolge und Tardiness
    for j, job in enumerate(jobs):
        seq = all_ops[j]
        num_ops = len(seq)
    
        # Technologische Reihenfolge: jede OP nach der vorherigen
        for o in range(1, num_ops):
            duration_prev = seq[o - 1][2]
            prob += starts[(j, o)] >= starts[(j, o - 1)] + duration_prev
    
        # Endzeit = letzte Startzeit + Dauer der letzten OP
        duration_last = seq[-1][2]
        prob += ends[j] == starts[(j, num_ops - 1)] + duration_last
    
        # Tardiness ≥ Endzeit - Deadline
        prob += tard[j] >= ends[j] - deadline[job]


    # Maschinenkonflikte
    for m in machines:
        ops_on_m = [
            (j, o, seq[o][2])
            for j, seq in enumerate(all_ops)
            for o in range(len(seq))
            if seq[o][1] == m
        ]
        for i in range(len(ops_on_m)):
            j1, o1, d1 = ops_on_m[i]
            for j2, o2, d2 in ops_on_m[i + 1:]:
                if j1 == j2:
                    continue
                y = pulp.LpVariable(f"y_{j1}_{o1}_{j2}_{o2}", cat="Binary")
                prob += starts[(j1, o1)] + d1 + epsilon <= starts[(j2, o2)] + bigM * (1 - y)
                prob += starts[(j2, o2)] + d2 + epsilon <= starts[(j1, o1)] + bigM * y

    # Lösen
    solver = pulp.HiGHS_CMD(msg=msg_print, timeLimit=solver_time_limit, threads=threads)
    prob.solve(solver)

    total_tardiness = pulp.value(prob.objective)

    # Ergebnis extrahieren
    records = []
    for j, job in enumerate(jobs):
        for o, (op_id, m, d) in enumerate(all_ops[j]):
            st = starts[(j, o)].varValue
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

    df_schedule = (
        pd.DataFrame.from_records(records)
        .sort_values(["Arrival", "Start", "Job"])
        .reset_index(drop=True)
    )

    # Log
    print("\nSolver-Informationen:")
    print(f"  Zielfunktionswert       : {round(total_tardiness, 4)}")
    print(f"  Solver-Status           : {pulp.LpStatus[prob.status]}")
    print(f"  Anzahl Variablen        : {len(prob.variables())}")
    print(f"  Anzahl Constraints      : {len(prob.constraints)}")

    return df_schedule



# MAX

def solve_jssp_max_tardiness(
    df_jssp: pd.DataFrame,
    df_arrivals_deadlines: pd.DataFrame,
    solver_time_limit: int = 1200,
    epsilon: float = 0.0
) -> pd.DataFrame:
    """
    Minimiert die maximale Tardiness (Verspätung) unter allen Jobs.
    Zielfunktion: max_j [ max(0, Endzeit_j - Deadline_j) ]

    Parameter:
    - df_jssp: DataFrame mit ['Job','Operation','Machine','Processing Time'].
    - df_arrivals_deadlines: DataFrame mit ['Job','Arrival','Deadline'].
    - solver_time_limit: Max. Zeit in Sekunden für den Solver.
    - epsilon: Puffer zwischen Operationen auf derselben Maschine.

    Rückgabe:
    - df_schedule: DataFrame mit ['Job','Operation','Arrival','Deadline',
        'Machine','Start','Processing Time','End','Tardiness'].
    - max_tardiness: Maximale Verspätung (float).
    """
    # 1) Vorverarbeitung
    df_arr = df_arrivals_deadlines.sort_values("Arrival").reset_index(drop=True)
    arrival = df_arr.set_index("Job")["Arrival"].to_dict()
    deadline = df_arr.set_index("Job")["Deadline"].to_dict()
    jobs = df_arr["Job"].tolist()

    # 2) Operationen pro Job aus df_jssp aufbereiten
    ops_grouped = df_jssp.sort_values(["Job","Operation"]).groupby("Job")
    all_ops = []   # all_ops[j] = list of (op_id, machine_id, duration)
    machines = set()
    for job in jobs:
        seq = []
        grp = ops_grouped.get_group(job)
        for _, row in grp.iterrows():
            op_id = row["Operation"]
            m = int(re.search(r"M(\d+)", str(row["Machine"])).group(1))
            d = float(row["Processing Time"])
            seq.append((op_id, m, d))
            machines.add(m)
        all_ops.append(seq)

    n = len(jobs)
    bigM = 1e6

    # 3) LP-Problem definieren
    prob = pulp.LpProblem("JSSP_Max_Tardiness", pulp.LpMinimize)

    # 3a) Variablen
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
    max_tardiness = pulp.LpVariable("max_tardiness", lowBound=0)

    # 3b) Zielfunktion
    prob += max_tardiness

    # 4) Technologische Reihenfolge & Ankunft
    for j, job in enumerate(jobs):
        seq = all_ops[j]
        # erste Operation ≥ Arrival
        prob += starts[(j, 0)] >= arrival[job]
        # Folge-OPs
        for o in range(1, len(seq)):
            _, _, d_prev = seq[o-1]
            prob += starts[(j, o)] >= starts[(j, o-1)] + d_prev
        # Endzeit = letzte OP
        _, _, d_last = seq[-1]
        prob += ends[j] == starts[(j, len(seq)-1)] + d_last
        # Tardiness und max-Verknüpfung
        prob += tard[j] >= ends[j] - deadline[job]
        prob += max_tardiness >= tard[j]

    # 5) Maschinenkonflikte (Disjunktivität)
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

    # 6) Lösen
    prob.solve(pulp.HiGHS_CMD(msg=True, timeLimit=solver_time_limit))
    max_val = pulp.value(max_tardiness)

    # 7) Zeitplan extrahieren
    records = []
    for j, job in enumerate(jobs):
        for o, (op_id, m, d) in enumerate(all_ops[j]):
            st = starts[(j, o)].varValue
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

    df_schedule = (
        pd.DataFrame.from_records(records)
          .sort_values(["Arrival","Start","Job"])
          .reset_index(drop=True)
    )

    return df_schedule


# Lateness ---------------------------------------------------------------------------------------
def solve_jssp_max_absolute_lateness(
    df_jssp: pd.DataFrame,
    df_arrivals_deadlines: pd.DataFrame,
    solver_time_limit: int = 1200,
    epsilon: float = 0.0
) -> pd.DataFrame:
    """
    Minimiert die maximale absolute Lateness (Früh- oder Spätfertigung) über alle Jobs.
    Zielfunktion: max_j [ |C_j - d_j| ]

    Parameter:
    - df_jssp: DataFrame mit ['Job','Operation','Machine','Processing Time'].
    - df_arrivals_deadlines: DataFrame mit ['Job','Arrival','Deadline'].
    - solver_time_limit: Max. Zeit in Sekunden für den Solver.
    - epsilon: Abstand zur Vermeidung von Maschinenkonflikten.

    Rückgabe:
    - df_schedule: DataFrame mit ['Job','Operation','Arrival','Deadline',
        'Machine','Start','Processing Time','End','Lateness','Absolute Lateness'].
    - max_abs_lateness: Maximale absolute Lateness (float).
    """
    # 1) Vorverarbeitung
    df_arr = df_arrivals_deadlines.sort_values("Arrival").reset_index(drop=True)
    arrival = df_arr.set_index("Job")["Arrival"].to_dict()
    deadline = df_arr.set_index("Job")["Deadline"].to_dict()
    jobs = df_arr["Job"].tolist()

    # 2) Operationen pro Job aus df_jssp aufbereiten
    ops_grouped = df_jssp.sort_values(["Job","Operation"]).groupby("Job")
    all_ops = []   # all_ops[j] = list of (op_id, machine_id, duration)
    machines = set()
    for job in jobs:
        seq = []
        grp = ops_grouped.get_group(job)
        for _, row in grp.iterrows():
            op_id = row["Operation"]
            m = int(re.search(r"M(\d+)", str(row["Machine"])).group(1))
            d = float(row["Processing Time"])
            seq.append((op_id, m, d))
            machines.add(m)
        all_ops.append(seq)

    n = len(jobs)
    bigM = 1e6

    # 3) LP-Problem definieren
    prob = pulp.LpProblem("JSSP_Max_Absolute_Lateness", pulp.LpMinimize)

    # Variablen
    starts = {
        (j, o): pulp.LpVariable(f"start_{j}_{o}", lowBound=0)
        for j in range(n) for o in range(len(all_ops[j]))
    }
    ends = {
        j: pulp.LpVariable(f"end_{j}", lowBound=0)
        for j in range(n)
    }
    abs_lateness = {
        j: pulp.LpVariable(f"abs_lateness_{j}", lowBound=0)
        for j in range(n)
    }
    max_abs_lateness = pulp.LpVariable("max_abs_lateness", lowBound=0)

    # Zielfunktion
    prob += max_abs_lateness

    # 4) Reihenfolge & Ankunft
    for j, job in enumerate(jobs):
        seq = all_ops[j]
        prob += starts[(j, 0)] >= arrival[job]
        for o in range(1, len(seq)):
            _, _, d_prev = seq[o-1]
            prob += starts[(j, o)] >= starts[(j, o-1)] + d_prev
        # Ende der letzten Operation
        _, _, d_last = seq[-1]
        prob += ends[j] == starts[(j, len(seq)-1)] + d_last
        # Lateness ± und max-Verknüpfung
        lateness = ends[j] - deadline[job]
        prob += abs_lateness[j] >=  lateness
        prob += abs_lateness[j] >= -lateness
        prob += max_abs_lateness >= abs_lateness[j]

    # 5) Maschinenkonflikte
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

    # 6) Lösen
    prob.solve(pulp.HiGHS_CMD(msg=True, timeLimit=solver_time_limit))
    max_val = pulp.value(max_abs_lateness)

    # 7) Ergebnis extrahieren
    records = []
    for j, job in enumerate(jobs):
        for o, (op_id, m, d) in enumerate(all_ops[j]):
            st = starts[(j, o)].varValue
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
                "Lateness": round(ed - deadline[job], 2),
                "Absolute Lateness": abs(round(ed - deadline[job], 2))
            })

    df_schedule = (
        pd.DataFrame.from_records(records)
          .sort_values(["Arrival", "Start", "Job"])
          .reset_index(drop=True)
    )
    print("Maximale absolute Lateness:", round(max_val, 3))
    return df_schedule
