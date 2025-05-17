import pulp
import pandas as pd

def solve_jssp_individual_flowtime(df_jssp: pd.DataFrame,
                                   df_arrivals: pd.DataFrame,
                                   solver_time_limit: int = 300,
                                   epsilon: float = 0.0):
    """
    Minimiert die Summe der individuellen Durchlaufzeiten (Flow Times) aller Jobs.
    Zielfunktion: sum_j (Endzeit_j - Ankunftszeit_j)

    Parameter:
    - df_jssp: DataFrame mit ['Job','Operation','Machine','Processing Time'].
    - df_arrivals: DataFrame mit ['Job','Arrival'].
    - solver_time_limit: Max. Laufzeit für HiGHS (Sekunden).
    - epsilon: Puffer zwischen Operationen auf derselben Maschine (gleiche Einheit wie Processing Time).

    Rückgabe:
    - df_schedule: DataFrame mit ['Job','Operation','Arrival','Machine','Start',
      'Processing Time','Flow time','End']
    - total_flowtime: Summe aller Flow Times (float)
    """
    # 1) Arrival-Dict
    arrival = df_arrivals.set_index('Job')['Arrival'].to_dict()

    # 2) Operationen DataFrame
    df = df_jssp.copy()

    # 3) LP-Modell initialisieren
    prob = pulp.LpProblem('JSSP_IndFlow', pulp.LpMinimize)

    # 4) Startzeiten-Variablen pro Operation
    starts = {
        (row.Job, row.Operation): pulp.LpVariable(f"start_{row.Job}_{row.Operation}", lowBound=0)
        for row in df.itertuples()
    }
    # Endzeiten-Variable pro Job
    job_ends = {
        job: pulp.LpVariable(f"end_{job}", lowBound=0)
        for job in df['Job'].unique()
    }

    # 5) Zielfunktion
    prob += pulp.lpSum(job_ends[job] - arrival[job] for job in job_ends)

    # 6) Technologische Reihenfolge & Ankunft
    for job, grp in df.groupby('Job', sort=False):
        seq = grp.sort_values('Operation')
        # erste Operation ≥ Arrival
        first = seq.iloc[0]
        prob += starts[(job, first.Operation)] >= arrival[job]
        # Reihenfolgebedingungen
        for prev, curr in zip(seq.itertuples(), seq.iloc[1:].itertuples()):
            prob += (starts[(curr.Job, curr.Operation)]
                     >= starts[(prev.Job, prev.Operation)] + prev._4)
        # letzte Operation setzt Endzeit
        last = seq.iloc[-1]
        prob += job_ends[job] >= starts[(job, last.Operation)] + last._4

    # 7) Maschinenkonflikte per groupby
    M = 1e6
    for machine, grp in df.groupby('Machine', sort=False):
        group = grp.reset_index(drop=True)
        for i in range(len(group)):
            for j in range(i+1, len(group)):
                row_i = group.iloc[i]
                row_j = group.iloc[j]
                # nur zwischen unterschiedlichen Jobs
                if row_i.Job == row_j.Job:
                    continue
                s_i = starts[(row_i.Job, row_i.Operation)]
                s_j = starts[(row_j.Job, row_j.Operation)]
                d_i = row_i._4
                d_j = row_j._4
                y = pulp.LpVariable(f"y_{row_i.Job}_{row_i.Operation}_{row_j.Job}_{row_j.Operation}", cat='Binary')
                prob += s_i + d_i + epsilon <= s_j + M*(1-y)
                prob += s_j + d_j + epsilon <= s_i + M*y

    # 8) Lösen
    prob.solve(pulp.HiGHS_CMD(msg=True, timeLimit=solver_time_limit))

    # 9) Ergebnis extrahieren
    recs = []
    for r in df.itertuples():
        st = starts[(r.Job, r.Operation)].varValue
        end = st + r._4
        recs.append({
            'Job': r.Job,
            'Operation': r.Operation,
            'Arrival': arrival[r.Job],
            'Machine': r.Machine,
            'Start': round(st,2),
            'Processing Time': r._4,
            'Flow time': round(end - arrival[r.Job],2),
            'End': round(end,2)
        })
    df_schedule = pd.DataFrame(recs).sort_values(['Arrival','Start']).reset_index(drop=True)
    total_flowtime = round(pulp.value(prob.objective),3)

    return df_schedule, total_flowtime
