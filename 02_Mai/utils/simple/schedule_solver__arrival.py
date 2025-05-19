# global_makespan, fcfs

import pulp
import pandas as pd

from collections import defaultdict


def solve_jssp_global_makespan(df_jssp: pd.DataFrame,
                               df_arrivals: pd.DataFrame,
                               solver_time_limit: int = 300,
                               epsilon: float = 0.06):
    """
    Minimiert den Makespan eines Job-Shop-Problems mit Ankunftszeiten.

    Parameter:
    - df_jssp: DataFrame mit ['Job','Operation','Machine','Processing Time'],
               Operationen in sequentieller Reihenfolge je Job.
    - df_arrivals: DataFrame mit ['Job','Arrival'].
    - solver_time_limit: Max. Laufzeit für HiGHS (Sekunden).
    - epsilon: Puffer (Minuten) zwischen Operationen auf derselben Maschine.

    Rückgabe:
    - df_schedule: DataFrame mit ['Job','Operation','Arrival','Machine','Start',
      'Processing Time','Flow time','End']
    - makespan_value: minimaler Makespan (float)
    """
    # 1) Ankunftszeiten
    df_arr = df_arrivals.sort_values('Arrival').reset_index(drop=True)
    arrival = df_arr.set_index('Job')['Arrival'].to_dict()

    # 2) Job-Reihenfolge: absteigend nach Arrival
    job_order = df_arr.sort_values('Arrival', ascending=False)['Job'].tolist()

    # 3) Operationen gruppieren
    ops_grouped = {
        job: grp.sort_values('Operation')[['Operation','Machine','Processing Time']]
                 .apply(lambda r: (int(r['Operation']), int(r['Machine'].lstrip('M')), r['Processing Time']), axis=1)
                 .tolist()
        for job, grp in df_jssp.groupby('Job', sort=False)
        if job in arrival
    }
    num_jobs = len(job_order)
    all_ops = [ops_grouped[j] for j in job_order]

    # 4) Maschinenliste
    machines = df_jssp['Machine'].str.lstrip('M').astype(int).unique().tolist()

    # 5) LP-Modell
    prob = pulp.LpProblem('JSSP_Global_Makespan', pulp.LpMinimize)
    # Startvariablen
    starts = {
        (j,op): pulp.LpVariable(f'start_{j}_{op}', lowBound=0)
        for j in range(num_jobs) for op,_,_ in all_ops[j]
    }
    makespan = pulp.LpVariable('makespan', lowBound=0)
    prob += makespan

    # 6) Reihenfolge & Ankunftszeit
    for j, job in enumerate(job_order):
        ops = all_ops[j]
        prob += starts[(j, ops[0][0])] >= arrival[job]
        for _, op, dur in ops:
            # find next op in sequence
            pass  # handled by following loop
        for i in range(len(ops)-1):
            cur_op, _, cur_dur = ops[i]
            next_op, _, _ = ops[i+1]
            prob += starts[(j, next_op)] >= starts[(j, cur_op)] + cur_dur

    # 7) Maschinenkonflikte
    M = 1e5
    for m in machines:
        on_m = [(j, op, dur) for j in range(num_jobs)
                for op, mach, dur in all_ops[j] if mach == m]
        for idx,(j1,o1,d1) in enumerate(on_m):
            for j2,o2,d2 in on_m[idx+1:]:
                if j1 == j2: continue
                y = pulp.LpVariable(f'y_{j1}_{o1}_{j2}_{o2}', cat='Binary')
                prob += starts[(j1,o1)] + d1 + epsilon <= starts[(j2,o2)] + M*(1-y)
                prob += starts[(j2,o2)] + d2 + epsilon <= starts[(j1,o1)] + M*y

    # 8) Makespan-Bedingungen
    for j in range(num_jobs):
        last_op = all_ops[j][-1][0]
        last_dur = all_ops[j][-1][2]
        prob += makespan >= starts[(j, last_op)] + last_dur

    # 9) Lösen
    solver = pulp.HiGHS_CMD(msg=True, timeLimit=solver_time_limit)
    prob.solve(solver)

    # 10) Ergebnis extrahieren
    recs = []
    for j in range(num_jobs):
        for op, mach, dur in all_ops[j]:
            st = starts[(j,op)].varValue
            end = st + dur
            job = job_order[j]
            recs.append({
                'Job': job,
                'Operation': op,
                'Arrival': arrival[job],
                'Machine': f'M{mach}',
                'Start': round(st,2),
                'Processing Time': dur,
                'Flow time': round(end - arrival[job],2),
                'End': round(end,2)
            })
    df_schedule = pd.DataFrame(recs)
    # 11) Sortieren
    df_schedule = df_schedule.sort_values(['Arrival','Start']).reset_index(drop=True)

    makespan_value = round(pulp.value(makespan),3)
    return df_schedule, makespan_value



def schedule_fcfs_with_arrivals(df_jssp: pd.DataFrame,
                                arrival_df: pd.DataFrame) -> pd.DataFrame:
    """
    FCFS-Scheduling mit Job-Ankunftszeiten auf Basis eines DataFrames.

    Parameter:
    - df_jssp: DataFrame mit ['Job','Operation','Machine','Processing Time'].
    - arrival_df: DataFrame mit ['Job','Arrival'].

    Rückgabe:
    - df_schedule: DataFrame mit ['Job','Operation','Arrival','Machine',
      'Start','Processing Time','End'], sortiert nach ['Arrival','Start'].
    """
    # Arrival-Zeiten als Dict
    arrival = arrival_df.set_index('Job')['Arrival'].to_dict()

    # Status-Tracker
    next_op = {job: 0 for job in df_jssp['Job'].unique()}
    job_ready = arrival.copy()
    machine_ready = defaultdict(float)
    remaining = len(df_jssp)

    schedule = []
    while remaining > 0:
        best = None  # (job, start, dur, machine, op_idx)

        # Suche FCFS-geeignete Operation
        for job, op_idx in next_op.items():
            # Skip, wenn alle Ops geplant
            if op_idx >= (df_jssp['Job'] == job).sum():
                continue
            # Hole Row anhand Job+Operation
            row = df_jssp[(df_jssp['Job']==job)&(df_jssp['Operation']==op_idx)].iloc[0]
            m = int(row['Machine'].lstrip('M'))
            dur = row['Processing Time']
            earliest = max(job_ready[job], machine_ready[m])
            # Best-Kandidat wählen
            if (best is None or
                earliest < best[1] or
                (earliest == best[1] and arrival[job] < arrival[best[0]])):
                best = (job, earliest, dur, m, op_idx)

        job, start, dur, m, op_idx = best
        end = start + dur
        schedule.append({
            'Job': job,
            'Operation': op_idx,
            'Arrival': arrival[job],
            'Machine': f'M{m}',
            'Start': start,
            'Processing Time': dur,
            'End': end
        })
        # Update Status
        job_ready[job] = end
        machine_ready[m] = end
        next_op[job] += 1
        remaining -= 1

    df_schedule = pd.DataFrame(schedule)
    return df_schedule.sort_values(['Arrival','Start'])
