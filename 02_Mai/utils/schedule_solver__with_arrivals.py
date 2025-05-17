import pulp
import pandas as pd

import pulp
import pandas as pd

import pulp
import pandas as pd

def solve_jssp_individual_flowtime(df_jssp: pd.DataFrame,
                                   df_arrivals: pd.DataFrame,
                                   solver_time_limit: int = 300,
                                   epsilon: float = 0.0):
    """
    Spiegelung der dict-basierten Lösung für individuelle Flow Times,
    jetzt auf DataFrames übertragen.

    1) Sortiere df_arrivals nach Arrival aufsteigend, speichere es
       und erzeuge job_order absteigend.
    2) Baue all_ops analog zum dict auf.
    3) Setze Constraints exakt wie vorher.
    """

    # --- 1) Arrival und Job-Ordering ----------------------------
    # (a) aufsteigend sortieren, damit mapping konsistent ist
    df_arr = df_arrivals.sort_values("Arrival").reset_index(drop=True)
    arrival = df_arr.set_index("Job")["Arrival"].to_dict()
    # (b) für die LP-Variable-Erzeugung und Constraints: absteigend
    job_order = df_arr.sort_values("Arrival", ascending=False)["Job"].tolist()

    num_jobs = len(job_order)

    # --- 2) all_ops wie im dict -------------------------------
    # ops_grouped[job] = [(machine_idx, duration), ...] in Operation-Reihenfolge
    ops_grouped = {
        job: grp.sort_values("Operation")[["Machine","Processing Time"]]
                  .apply(lambda r: (int(r["Machine"].lstrip("M")), r["Processing Time"]),
                         axis=1)
                  .tolist()
        for job, grp in df_jssp.groupby("Job", sort=False)
    }
    all_ops = [ops_grouped[j] for j in job_order]
    all_machines = {m for ops in all_ops for m,_ in ops}

    # --- 3) LP aufsetzen ---------------------------------------
    prob = pulp.LpProblem("JSSP_IndFlow_DICT_EQUIV", pulp.LpMinimize)

    # Startvariablen (j,o)
    starts = {
        (j, o): pulp.LpVariable(f"start_{j}_{o}", lowBound=0)
        for j in range(num_jobs)
        for o in range(len(all_ops[j]))
    }
    # Job-Endzeiten
    job_ends = {
        j: pulp.LpVariable(f"end_{j}", lowBound=0)
        for j in range(num_jobs)
    }

    # Objective = Sum(end_j - arrival[job])
    prob += pulp.lpSum(
        job_ends[j] - arrival[job_order[j]]
        for j in range(num_jobs)
    )

    # Technologische Reihenfolge + Ankunft
    for j, job in enumerate(job_order):
        ops = all_ops[j]
        # erste Op ≥ arrival
        prob += starts[(j, 0)] >= arrival[job]
        # Folge-OPs
        for o in range(1, len(ops)):
            dur_prev = ops[o-1][1]
            prob += starts[(j, o)] >= starts[(j, o-1)] + dur_prev
        # letzte Operation setzt job_end
        last_o, last_d = len(ops)-1, ops[-1][1]
        prob += job_ends[j] >= starts[(j, last_o)] + last_d

    # Maschinenkonflikte
    bigM = 1e6
    for m in all_machines:
        # Alle Operationen auf Maschine m
        ops_on_m = [
            (j, o, d)
            for j in range(num_jobs)
            for o, (mach, d) in enumerate(all_ops[j])
            if mach == m
        ]
        for idx in range(len(ops_on_m)):
            j1,o1,d1 = ops_on_m[idx]
            for j2,o2,d2 in ops_on_m[idx+1:]:
                if j1 == j2: continue
                y = pulp.LpVariable(f"y_{j1}_{o1}_{j2}_{o2}", cat="Binary")
                prob += starts[(j1,o1)] + d1 + epsilon <= starts[(j2,o2)] + bigM*(1-y)
                prob += starts[(j2,o2)] + d2 + epsilon <= starts[(j1,o1)] + bigM*y

    # Lösen
    prob.solve(pulp.HiGHS_CMD(msg=True, timeLimit=solver_time_limit))

    # --- 4) Ergebnis extrahieren und exakte Spaltenreihenfolge ---
    records = []
    for j, job in enumerate(job_order):
        for o, (mach, dur) in enumerate(all_ops[j]):
            st = starts[(j,o)].varValue
            ed = st + dur
            records.append({
                "Job": job,
                "Operation": o,
                "Arrival": arrival[job],          # korrekte Ankunft
                "Machine": f"M{mach}",
                "Start": round(st, 2),
                "Processing Time": dur,
                "Flow time": round(ed - arrival[job], 2),
                "End": round(ed, 2)
            })

    df_schedule = (
        pd.DataFrame(records)
          .sort_values(["Arrival","Start"])
          .reset_index(drop=True)
    )
    total_flowtime = round(pulp.value(prob.objective), 3)
    return df_schedule, total_flowtime





def solve_jssp_weighted_individual_flowtime(df_jssp: pd.DataFrame,
                                            df_arrivals: pd.DataFrame,
                                            solver_time_limit: int = 300,
                                            epsilon: float = 0.0):
    """
    Minimiert die gewichtete Summe der individuellen Durchlaufzeiten aller Jobs.
    Gewicht_j = 1 / (1 + Arrival_j)
    Zielfunktion: sum_j Gewicht_j * (Endzeit_j - Arrival_j)

    Parameter:
    - df_jssp: DataFrame mit ['Job','Operation','Machine','Processing Time'].
    - df_arrivals: DataFrame mit ['Job','Arrival'].
    - solver_time_limit: Max. Laufzeit für HiGHS (Sekunden).
    - epsilon: Puffer zwischen Operationen auf derselben Maschine.

    Rückgabe:
    - df_schedule: DataFrame mit ['Job','Operation','Arrival','Machine',
      'Start','Processing Time','Flow time','End'], sortiert nach ['Arrival','Start'].
    - total_weighted_flowtime: gewichtete Summe der Durchlaufzeiten (float).
    """
    # Arrival lookup and weights
    arrival = df_arrivals.set_index('Job')['Arrival'].to_dict()
    weights = {job: 1.0 / (1.0 + arrival[job]) for job in arrival}

    df = df_jssp.copy()
    jobs = df['Job'].unique().tolist()

    # Initialize model
    prob = pulp.LpProblem('JSSP_Weighted_IndFlow', pulp.LpMinimize)

    # Start-time variables
    starts = {
        (row.Job, row.Operation): pulp.LpVariable(f"start_{row.Job}_{row.Operation}", lowBound=0)
        for row in df.itertuples(index=False)
    }
    # End-time per job
    job_end = {
        job: pulp.LpVariable(f"end_{job}", lowBound=0)
        for job in jobs
    }

    # Objective: weighted sum of (end - arrival)
    prob += pulp.lpSum(weights[j] * (job_end[j] - arrival[j]) for j in jobs)

    # Tech order + arrival
    for job, grp in df.groupby('Job', sort=False):
        seq = grp.sort_values('Operation').reset_index(drop=True)
        # first op ≥ arrival
        prob += starts[(job, seq.loc[0, 'Operation'])] >= arrival[job]
        # sequence constraints
        for i in range(len(seq) - 1):
            op_i = seq.loc[i, 'Operation']
            dur_i = seq.loc[i, 'Processing Time']
            op_n = seq.loc[i+1, 'Operation']
            prob += starts[(job, op_n)] >= starts[(job, op_i)] + dur_i
        # link last op to job_end
        last_op = seq.loc[len(seq)-1, 'Operation']
        last_dur = seq.loc[len(seq)-1, 'Processing Time']
        prob += job_end[job] == starts[(job, last_op)] + last_dur

    # Machine conflicts
    M = 1e6
    for machine, grp in df.groupby('Machine', sort=False):
        block = grp.reset_index(drop=True)
        for i in range(len(block)):
            for j in range(i+1, len(block)):
                ri = block.loc[i]
                rj = block.loc[j]
                if ri.Job == rj.Job:
                    continue
                si = starts[(ri.Job, ri.Operation)]
                sj = starts[(rj.Job, rj.Operation)]
                di = ri['Processing Time']
                dj = rj['Processing Time']
                y = pulp.LpVariable(f"y_{ri.Job}_{ri.Operation}_{rj.Job}_{rj.Operation}", cat='Binary')
                prob += si + di + epsilon <= sj + M*(1-y)
                prob += sj + dj + epsilon <= si + M*y

    # Solve
    prob.solve(pulp.HiGHS_CMD(msg=True, timeLimit=solver_time_limit))

    # Extract schedule
    recs = []
    for _, row in df.iterrows():
        st = starts[(row.Job, row.Operation)].varValue
        end = st + row['Processing Time']
        recs.append({
            'Job': row.Job,
            'Operation': row.Operation,
            'Arrival': arrival[row.Job],
            'Machine': row.Machine,
            'Start': round(st, 2),
            'Processing Time': row['Processing Time'],
            'Flow time': round(end - arrival[row.Job], 2),
            'End': round(end, 2)
        })

    df_schedule = pd.DataFrame(recs) \
        .sort_values(['Arrival','Start']) \
        .reset_index(drop=True)

    total_weighted_flowtime = round(pulp.value(prob.objective), 3)
    return df_schedule, total_weighted_flowtime

