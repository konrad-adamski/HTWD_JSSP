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

