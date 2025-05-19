import pulp
import pandas as pd

def solve_jssp_global_makespan(df_jssp: pd.DataFrame,
                      df_arrivals: pd.DataFrame,
                      solver: str = 'HiGHS',
                      solver_time_limit: int = 300,
                      epsilon: float = 0.06):
    """
    Optimierte Job-Shop-Scheduling-Funktion mit Ankunftszeiten, exakter Reihenfolge,
    Maschinenkonfliktauflösung und flexibler Solver-Wahl.

    Parameter:
    - df_jssp: DataFrame mit ['Job','Operation','Machine','Processing Time']
    - df_arrivals: DataFrame mit ['Job','Arrival']
    - solver: 'HiGHS' oder 'CBC'
    - solver_time_limit: Zeitlimit in Sekunden
    - epsilon: Pufferzeit zwischen Operationen auf derselben Maschine

    Rückgabe:
    - df_schedule: Zeitplan mit ['Job','Operation','Arrival','Machine','Start',
      'Processing Time','Flow time','End']
    - makespan_value: Wert des minimalen Makespans
    """

    # 1. Vorbereitung der Ankunftszeiten
    df_arr = df_arrivals.sort_values('Arrival').reset_index(drop=True)
    arrival = df_arr.set_index('Job')['Arrival'].to_dict()
    job_order = df_arr['Job'].tolist()
    num_jobs = len(job_order)

    # 2. Operationen gruppieren nach Job
    ops_grouped = {
        job: grp.sort_values('Operation')[['Operation', 'Machine', 'Processing Time']]
               .apply(lambda r: (int(r['Operation']), r['Machine'], r['Processing Time']), axis=1)
               .tolist()
        for job, grp in df_jssp.groupby('Job', sort=False)
        if job in arrival
    }
    all_ops = [ops_grouped[job] for job in job_order]

    # 3. Alle Maschinen extrahieren
    machines = df_jssp['Machine'].unique().tolist()

    # 4. LP-Modell definieren
    prob = pulp.LpProblem('JSSP_BestOf', pulp.LpMinimize)
    starts = {
        (j, op): pulp.LpVariable(f'start_{j}_{op}', lowBound=0)
        for j in range(num_jobs) for op, _, _ in all_ops[j]
    }
    makespan = pulp.LpVariable('makespan', lowBound=0)
    prob += makespan

    # 5. Technologische Reihenfolge und Ankunftszeiten
    for j, job in enumerate(job_order):
        ops = all_ops[j]
        prob += starts[(j, ops[0][0])] >= arrival[job]
        for i in range(len(ops) - 1):
            op1, _, dur1 = ops[i]
            op2, _, _ = ops[i + 1]
            prob += starts[(j, op2)] >= starts[(j, op1)] + dur1

    # 6. Maschinenkonflikte
    M = 1e6
    for m in machines:
        on_machine = [
            (j, op, dur) for j in range(num_jobs)
            for op, mach, dur in all_ops[j] if mach == m
        ]
        for idx1 in range(len(on_machine)):
            j1, o1, d1 = on_machine[idx1]
            for j2, o2, d2 in on_machine[idx1 + 1:]:
                if j1 == j2:
                    continue
                y = pulp.LpVariable(f'y_{j1}_{o1}_{j2}_{o2}', cat='Binary')
                prob += starts[(j1, o1)] + d1 + epsilon <= starts[(j2, o2)] + M * (1 - y)
                prob += starts[(j2, o2)] + d2 + epsilon <= starts[(j1, o1)] + M * y

    # 7. Makespan-Definition
    for j in range(num_jobs):
        last_op = all_ops[j][-1]
        prob += makespan >= starts[(j, last_op[0])] + last_op[2]

    # 8. Solver auswählen und lösen
    solver = solver.upper()
    if solver == 'HIGHS':
        cmd = pulp.HiGHS_CMD(msg=True, timeLimit=solver_time_limit)
    elif solver == 'CBC':
        cmd = pulp.PULP_CBC_CMD(msg=True, timeLimit=solver_time_limit)
    else:
        raise ValueError("Solver must be 'HiGHS' or 'CBC'")
    prob.solve(cmd)

    # 9. Ergebnis extrahieren
    records = []
    for j in range(num_jobs):
        for op, mach, dur in all_ops[j]:
            st = starts[(j, op)].varValue
            end = st + dur
            job = job_order[j]
            records.append({
                'Job': job,
                'Operation': op,
                'Arrival': arrival[job],
                'Machine': mach,
                'Start': round(st, 2),
                'Processing Time': dur,
                'Flow time': round(end - arrival[job], 2),
                'End': round(end, 2)
            })
    df_schedule = pd.DataFrame(records).sort_values(['Start', 'Job', 'Operation']).reset_index(drop=True)
    makespan_value = round(pulp.value(makespan), 3)

    return df_schedule


def solve_jssp_individual_flowtime(df_jssp: pd.DataFrame,
                                   df_arrivals: pd.DataFrame,
                                   solver_time_limit: int = 300,
                                   epsilon: float = 0.0):
    """
    Minimiert die Summe der individuellen Durchlaufzeiten (Flow Times) aller Jobs.
    Zielfunktion: sum_j (Endzeit_j - Arrival_j)

    Parameter:
    - df_jssp: DataFrame mit ['Job','Operation','Machine','Processing Time'].
    - df_arrivals: DataFrame mit ['Job','Arrival'].
    - solver_time_limit: Max. Laufzeit für HiGHS (Sekunden).
    - epsilon: Puffer zwischen Operationen auf derselben Maschine.

    Rückgabe:
    - df_schedule: DataFrame mit ['Job','Operation','Arrival','Machine',
      'Start','Processing Time','Flow time','End'].
    - total_flowtime: Summe aller Flow Times (float).
    """
    # Arrival-Lookup
    arrival = df_arrivals.set_index('Job')['Arrival'].to_dict()
    df = df_jssp.copy()

    # Modell
    prob = pulp.LpProblem('JSSP_IndFlow', pulp.LpMinimize)

    # Variablen
    starts = {
        (row.Job, row.Operation): pulp.LpVariable(f"start_{row.Job}_{row.Operation}", lowBound=0)
        for row in df.itertuples(index=False)
    }
    job_ends = {
        job: pulp.LpVariable(f"end_{job}", lowBound=0)
        for job in df['Job'].unique()
    }

    # Objective
    prob += pulp.lpSum(job_ends[j] - arrival[j] for j in job_ends)

    # Technologische Reihenfolge + Arrival
    for job, grp in df.groupby('Job', sort=False):
        seq = grp.sort_values('Operation').reset_index(drop=True)
        # erste Operation ≥ arrival
        prob += starts[(job, seq.loc[0, 'Operation'])] >= arrival[job]
        # Folge-OPs
        for i in range(len(seq)-1):
            op_prev = seq.loc[i, 'Operation']
            dur_prev = seq.loc[i, 'Processing Time']
            op_next = seq.loc[i+1, 'Operation']
            prob += starts[(job, op_next)] >= starts[(job, op_prev)] + dur_prev
        # letzte OP → job_end
        last_op = seq.loc[len(seq)-1, 'Operation']
        dur_last = seq.loc[len(seq)-1, 'Processing Time']
        prob += job_ends[job] >= starts[(job, last_op)] + dur_last

    # Maschinenkonflikte
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

    # Lösen
    prob.solve(pulp.HiGHS_CMD(msg=True, timeLimit=solver_time_limit))

    # Ergebnis
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

    df_schedule = pd.DataFrame(recs).sort_values(['Arrival','Start']).reset_index(drop=True)
    total_flowtime = round(pulp.value(prob.objective), 3)
    return df_schedule




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
    return df_schedule

