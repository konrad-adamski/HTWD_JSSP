import pulp
import pandas as pd


def solve_jobshop(df_jssp: pd.DataFrame,
                  solver: str = 'CBC',
                  solver_time_limit: int = 300,
                  epsilon: float = 0.0):
    """
    Minimiert den Makespan eines Job-Shop-Problems auf Basis eines DataFrames.

    Parameter:
    - df_jssp: DataFrame mit Spalten ['Job','Operation','Machine','Processing Time'].
    - solver:  'CBC' oder 'HiGHS' (wenigstens eine dieser beiden Optionen).
    - solver_time_limit: Zeitlimit in Sekunden.
    - epsilon: Puffer in Minuten zwischen Operationen auf derselben Maschine.

    Rückgabe:
    - df_schedule: DataFrame mit Spalten
      ['Job','Operation','Machine','Start','Processing Time','End']
    - makespan_value: minimaler Makespan (float)
    """
    # 1) Index und Op-Nummer
    df = df_jssp.reset_index(drop=False).rename(columns={'index':'Idx'}).copy()
    df['OpIdx'] = df.groupby('Job')['Operation'].first()  # nur zur Konsistenz

    # 2) LP-Modell
    prob = pulp.LpProblem('JSSP', pulp.LpMinimize)
    starts = {
        idx: pulp.LpVariable(f'start_{idx}', lowBound=0, cat='Continuous')
        for idx in df['Idx']
    }
    makespan = pulp.LpVariable('makespan', lowBound=0, cat='Continuous')
    prob += makespan

    # 3) Reihenfolge je Job
    for job, group in df.groupby('Job', sort=False):
        seq = group.sort_values('Operation')
        for prev, curr in zip(seq['Idx'], seq['Idx'][1:]):
            dur_prev = df.loc[df['Idx'] == prev, 'Processing Time'].iat[0]
            prob += starts[curr] >= starts[prev] + dur_prev

    # 4) Maschinenkonflikte
    M = 1e6
    for _, group in df.groupby('Machine', sort=False):
        ids = group['Idx'].tolist()
        for i in range(len(ids)):
            for j in range(i+1, len(ids)):
                i_idx, j_idx = ids[i], ids[j]
                if df.loc[df['Idx'] == i_idx, 'Job'].iat[0] == df.loc[df['Idx'] == j_idx, 'Job'].iat[0]:
                    continue
                di = df.loc[df['Idx'] == i_idx, 'Processing Time'].iat[0]
                dj = df.loc[df['Idx'] == j_idx, 'Processing Time'].iat[0]
                y = pulp.LpVariable(f'y_{i_idx}_{j_idx}', cat='Binary')
                prob += starts[i_idx] + di + epsilon <= starts[j_idx] + M*(1-y)
                prob += starts[j_idx] + dj + epsilon <= starts[i_idx] + M*y

    # 5) Makespan-Constraints
    for _, row in df.iterrows():
        idx = int(row['Idx'])
        prob += starts[idx] + row['Processing Time'] <= makespan

    # 6) Solver auswählen und ausführen
    if solver.upper() in ['CBC', 'BRANCH AND CUT']:
        cmd = pulp.PULP_CBC_CMD(msg=True, timeLimit=solver_time_limit)
    elif solver.upper() == 'HIGHS':
        cmd = pulp.HiGHS_CMD(msg=True, timeLimit=solver_time_limit)
    else:
        raise ValueError("Solver must be 'CBC' or 'HiGHS'")
    prob.solve(cmd)

    # 7) Ergebnis aufbereiten
    df['Start'] = df['Idx'].map(lambda i: round(starts[i].varValue, 2))
    df['End']   = df['Start'] + df['Processing Time']
    df_schedule = df[['Job','Operation','Machine','Start','Processing Time','End']]\
                  .sort_values(['Start','Job','Operation'])\
                  .reset_index(drop=True)
    makespan_value = round(pulp.value(makespan), 3)

    return df_schedule, makespan_value