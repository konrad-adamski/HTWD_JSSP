import pulp
import pandas as pd
import re

def solve_jssp_bi_criteria_flowtime_deviation_with_fixed_ops(
    df_jssp: pd.DataFrame,
    df_arrivals: pd.DataFrame,
    df_executed: pd.DataFrame,
    df_original_plan: pd.DataFrame,
    r: float = 0.5,
    solver_time_limit: int = 300,
    epsilon: float = 0.0,
    arrival_column: str = "Arrival",
    reschedule_start: float = 1440.0, 
    msg_print = False, threads= None
) -> pd.DataFrame:
    """
    Bi-kriterielle Rescheduling-Variante mit fixierten Operationen.
    Zielfunktion: Z(σ) = r * F(σ) + (1 - r) * D(σ)
    - F(σ): gewichtete individuelle Flow-Times
    - D(σ): Abweichung vom Originalplan

    Parameter:
    - df_jssp: DataFrame mit ['Job','Operation','Machine','Processing Time']
    - df_arrivals: DataFrame mit ['Job', arrival_column]
    - df_executed: DataFrame mit ['Job','Machine','Start','End']
    - df_original_plan: DataFrame mit ['Job','Operation','Start'] als Ursprungsplan
    - r: Gewichtung für Effizienz (F) vs. Stabilität (D)
    - solver_time_limit: max. Solverzeit in Sekunden
    - epsilon: Zeitpuffer zwischen Operationen auf gleicher Maschine
    - arrival_column: Name der Spalte mit Ankunftszeit
    - reschedule_start: Zeitpunkt, ab dem neu geplant wird

    Rückgabe:
    - df_schedule: DataFrame mit ['Job','Operation','Arrival','Machine','Start','Processing Time','Flow time','End']
    """

    # Vorbereitung
    df_arrivals = df_arrivals.sort_values(arrival_column).reset_index(drop=True)
    arrival_times = df_arrivals.set_index('Job')[arrival_column].to_dict()
    job_list = df_arrivals['Job'].tolist()
    num_jobs = len(job_list)
    weights = {job: 1.0 / (1.0 + arrival_times[job]) for job in job_list}

    # Originale Startzeiten extrahieren
    original_start_times = {
        (row['Job'], row['Operation']): row['Start']
        for _, row in df_original_plan.iterrows()
    }

    # Operationen & Maschinen
    ops_grouped = df_jssp.sort_values(['Job', 'Operation']).groupby('Job')
    all_ops, all_machines = [], set()
    for job in job_list:
        grp = ops_grouped.get_group(job)
        ops = []
        for _, row in grp.iterrows():
            op_id = row['Operation']
            mac_str = str(row['Machine'])
            m_id = int(re.search(r"M(\d+)", mac_str).group(1))
            dur = float(row['Processing Time'])
            ops.append((op_id, m_id, dur))
            all_machines.add(m_id)
        all_ops.append(ops)

    # Fixierte Operationen
    df_executed_fixed = df_executed[df_executed['End'] >= reschedule_start].copy()
    df_executed_fixed['MachineID'] = df_executed_fixed['Machine'].astype(str).str.extract(r"M(\d+)", expand=False).astype(int)
    fixed_ops = {
        m: list(gr[['Start', 'End', 'Job']].itertuples(index=False, name=None))
        for m, gr in df_executed_fixed.groupby('MachineID')
    }
    last_executed_end = df_executed.groupby('Job')['End'].max().to_dict()

    # LP-Modell
    prob = pulp.LpProblem('JSSP_BiCriteria_FlowDeviation_Fixed', pulp.LpMinimize)

    starts = {
        (j, o): pulp.LpVariable(f"start_{j}_{o}", lowBound=0)
        for j in range(num_jobs)
        for o in range(len(all_ops[j]))
    }
    job_ends = {
        j: pulp.LpVariable(f"job_end_{j}", lowBound=0)
        for j in range(num_jobs)
    }

    # Flow-Teil der Zielfunktion
    flow_efficiency = pulp.lpSum(
        weights[job_list[j]] * (job_ends[j] - arrival_times[job_list[j]])
        for j in range(num_jobs)
    )

    # Abweichungs-Teil der Zielfunktion
    deviations = {}
    for (j, o) in starts:
        job = job_list[j]
        key = (job, o)
        if key in original_start_times:
            dev = pulp.LpVariable(f"dev_{j}_{o}", lowBound=0)
            prob += dev >= starts[(j, o)] - original_start_times[key]
            prob += dev >= original_start_times[key] - starts[(j, o)]
            deviations[(j, o)] = dev
    deviation = pulp.lpSum(deviations.values())

    # Kombinierte Zielfunktion
    prob += r * flow_efficiency + (1 - r) * deviation

    # Technologische Reihenfolge + frühester Start
    for j, job in enumerate(job_list):
        earliest = max(arrival_times[job], last_executed_end.get(job, reschedule_start))
        prob += starts[(j, 0)] >= earliest
        for o in range(1, len(all_ops[j])):
            _, _, prev_dur = all_ops[j][o-1]
            prob += starts[(j, o)] >= starts[(j, o-1)] + prev_dur

    # Maschinenkonflikte
    M = 1e5
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
                y = pulp.LpVariable(f"y_{j1}_{o1}_{j2}_{o2}", cat='Binary')
                prob += (starts[(j1, o1)] + d1 + epsilon <= starts[(j2, o2)] + M*(1 - y))
                prob += (starts[(j2, o2)] + d2 + epsilon <= starts[(j1, o1)] + M*y)
        for j1, o1, d1 in ops_on_m:
            for fixed_start, fixed_end, fixed_job in fixed_ops.get(m, []):
                y_fix = pulp.LpVariable(f"y_fix_{j1}_{o1}_{fixed_job}", cat='Binary')
                prob += (starts[(j1, o1)] + d1 + epsilon <= fixed_start + M*(1 - y_fix))
                prob += (fixed_end + epsilon <= starts[(j1, o1)] + M*y_fix)

    # Endzeitbindung pro Job
    for j in range(num_jobs):
        last_o = len(all_ops[j]) - 1
        prob += job_ends[j] >= starts[(j, last_o)] + all_ops[j][last_o][2]

    # Solver starten
    if threads: 
        prob.solve(pulp.HiGHS_CMD(msg=msg_print, timeLimit=solver_time_limit, threads= threads))
    else:
        prob.solve(pulp.HiGHS_CMD(msg=msg_print, timeLimit=solver_time_limit))

    

    # Ergebnisse sammeln
    recs = []
    for (j, o), var in sorted(starts.items()):
        st = var.varValue
        if st is None:
            continue
        op_id, mach, dur = all_ops[j][o]
        end = st + dur
        recs.append({
            'Job': job_list[j],
            'Operation': op_id,
            'Arrival': arrival_times[job_list[j]],
            'Machine': f"M{mach}",
            'Start': round(st, 2),
            'Processing Time': dur,
            'Flow time': round(end - arrival_times[job_list[j]], 2),
            'End': round(end, 2)
        })

    df_schedule = pd.DataFrame(recs)
    df_schedule = df_schedule[['Job', 'Operation', 'Arrival', 'Machine', 'Start', 'Processing Time', 'Flow time', 'End']]
    df_schedule = df_schedule.sort_values(['Arrival', 'Start']).reset_index(drop=True)

    # Solver-Status & Informationen
    status = pulp.LpStatus[prob.status]
    objective_value = pulp.value(prob.objective)
    num_constraints = len(prob.constraints)
    num_variables = len(prob.variables())
    
    print("\nSolver-Informationen:")
    print(f"  Zielfunktionswert       : {round(objective_value, 4)}")
    print(f"  Solver-Status           : {status}")
    print(f"  Anzahl Variablen        : {num_variables}")
    print(f"  Anzahl Constraints      : {num_constraints}")
    return df_schedule



def solve_jssp_weighted_individual_flowtime_with_fixed_ops(
    df_jssp: pd.DataFrame,
    df_arrivals: pd.DataFrame,
    df_executed: pd.DataFrame,
    solver_time_limit: int = 300,
    epsilon: float = 0.0,
    arrival_column: str = "Arrival",
    reschedule_start: float = 1440.0
) -> pd.DataFrame:
    """
    Rescheduling-Variante mit fixierten Operationen.
    Minimiert die gewichtete Summe der individuellen Durchlaufzeiten aller Jobs.
    Gewicht_j = 1 / (1 + Arrival_j)
    Zielfunktion: sum_j Gewicht_j * (Endzeit_j - Arrival_j)

    Parameter:
    - df_jssp: DataFrame mit ['Job','Operation','Machine','Processing Time'].
    - df_arrivals: DataFrame mit ['Job', arrival_column].
    - df_executed: DataFrame mit ['Job','Machine','Start','End'] für bereits ausgeführte Ops.
    - solver_time_limit: Max. Laufzeit für HiGHS (Sekunden).
    - epsilon: Puffer zwischen Operationen auf derselben Maschine.
    - arrival_column: Name der Spalte in df_arrivals für Ankunftszeiten.
    - reschedule_start: Zeitpunkt, ab dem neu geplant wird.

    Rückgabe:
    - df_schedule: DataFrame mit ['Job','Operation','Arrival','Machine','Start','Processing Time','Flow time','End']
    """
    # Vorbereitung: Ankunftszeiten & Sortierung
    df_arrivals = df_arrivals.sort_values(arrival_column).reset_index(drop=True)
    arrival_times = df_arrivals.set_index('Job')[arrival_column].to_dict()
    job_list = df_arrivals['Job'].tolist()
    num_jobs = len(job_list)

    # Gewichtung berechnen
    weights = {job: 1.0 / (1.0 + arrival_times[job]) for job in job_list}

    # Job-Operationen aus df_jssp aufbereiten
    ops_grouped = df_jssp.sort_values(['Job', 'Operation']).groupby('Job')
    all_ops = []
    all_machines = set()
    for job in job_list:
        grp = ops_grouped.get_group(job)
        ops = []
        for _, row in grp.iterrows():
            op_id = row['Operation']
            mac_str = str(row['Machine'])
            match = re.search(r"M(\d+)", mac_str)
            m_id = int(match.group(1)) if match else 0
            dur = float(row['Processing Time'])
            ops.append((op_id, m_id, dur))
            all_machines.add(m_id)
        all_ops.append(ops)

    # Letzte Endzeit bereits geplanter Operationen pro Job
    last_executed_end = df_executed.groupby('Job')['End'].max().to_dict()

    # Filter fixierte Ops ab reschedule_start
    df_executed_fixed = df_executed[df_executed['End'] >= reschedule_start].copy()
    df_executed_fixed['MachineID'] = (
        df_executed_fixed['Machine'].astype(str)
        .str.extract(r"M(\d+)", expand=False)
        .astype(int)
    )
    fixed_ops = {
        m: list(gr[['Start', 'End', 'Job']].itertuples(index=False, name=None))
        for m, gr in df_executed_fixed.groupby('MachineID')
    }

    # LP-Modell
    prob = pulp.LpProblem('JSSP_WeightedIndFlow_Fixed', pulp.LpMinimize)

    # Variablen
    starts = {
        (j, o): pulp.LpVariable(f"start_{j}_{o}", lowBound=0)
        for j in range(num_jobs)
        for o in range(len(all_ops[j]))
    }
    job_ends = {
        j: pulp.LpVariable(f"job_end_{j}", lowBound=0)
        for j in range(num_jobs)
    }

    # Zielfunktion: gewichtete Summe (End - Arrival)
    prob += pulp.lpSum(
        weights[job_list[j]] * (job_ends[j] - arrival_times[job_list[j]])
        for j in range(num_jobs)
    )

    # Technologische Reihenfolge + individuelle Startbedingungen
    for j, job in enumerate(job_list):
        earliest = max(
            arrival_times[job],
            last_executed_end.get(job, reschedule_start)
        )
        prob += starts[(j, 0)] >= earliest
        for o in range(1, len(all_ops[j])):
            _, _, prev_dur = all_ops[j][o-1]
            prob += starts[(j, o)] >= starts[(j, o-1)] + prev_dur

    # Maschinenkonflikte
    M = 1e5
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
                y = pulp.LpVariable(f"y_{j1}_{o1}_{j2}_{o2}", cat='Binary')
                prob += (starts[(j1, o1)] + d1 + epsilon <= starts[(j2, o2)] + M*(1 - y))
                prob += (starts[(j2, o2)] + d2 + epsilon <= starts[(j1, o1)] + M*y)
        for j1, o1, d1 in ops_on_m:
            for fixed_start, fixed_end, fixed_job in fixed_ops.get(m, []):
                y_fix = pulp.LpVariable(f"y_fix_{j1}_{o1}_{fixed_job}", cat='Binary')
                prob += (starts[(j1, o1)] + d1 + epsilon <= fixed_start + M*(1 - y_fix))
                prob += (fixed_end + epsilon <= starts[(j1, o1)] + M*y_fix)

    # Endzeitbindung je Job
    for j in range(num_jobs):
        last_o = len(all_ops[j]) - 1
        prob += job_ends[j] >= starts[(j, last_o)] + all_ops[j][last_o][2]

    # Lösen
    prob.solve(pulp.HiGHS_CMD(msg=True, timeLimit=solver_time_limit))

    # Ergebnis extrahieren
    recs = []
    for (j, o), var in sorted(starts.items()):
        st = var.varValue
        if st is None:
            continue
        op_id, mach, dur = all_ops[j][o]
        end = st + dur
        recs.append({
            'Job': job_list[j],
            'Operation': op_id,
            'Arrival': arrival_times[job_list[j]],
            'Machine': f"M{mach}",
            'Start': round(st, 2),
            'Processing Time': dur,
            'Flow time': round(end - arrival_times[job_list[j]], 2),
            'End': round(end, 2)
        })

    df_schedule = pd.DataFrame(recs)
    cols = ['Job', 'Operation', 'Arrival', 'Machine', 'Start', 'Processing Time', 'Flow time', 'End']
    df_schedule = df_schedule[cols]
    df_schedule = df_schedule.sort_values(['Arrival', 'Start']).reset_index(drop=True)
    print("✅ Fertig!")
    return df_schedule



def solve_jssp_individual_flowtime_with_fixed_ops(
    df_jssp: pd.DataFrame,
    df_arrivals: pd.DataFrame,
    df_executed: pd.DataFrame,
    solver_time_limit: int = 300,
    epsilon: float = 0.0,
    arrival_column: str = "Arrival",
    reschedule_start: float = 1440.0
) -> pd.DataFrame:
    """
    Schnelle Rescheduling-Variante mit fixierten Operationen.
    Plant alle verbleibenden Jobs ab reschedule_start, unter Berücksichtigung der letzten
    Endzeit bereits ausgeführter Operationen pro Job.

    Minimiert die Summe der individuellen Durchlaufzeiten (Flow Times) aller Jobs.
    Zielfunktion: sum_j (Endzeit_j - Arrival_j)

    Parameter:
    - df_jssp: DataFrame mit ['Job','Operation','Machine','Processing Time'].
    - df_arrivals: DataFrame mit ['Job', arrival_column].
    - df_executed: DataFrame mit ['Job','Machine','Start','End'] für bereits ausgeführte Ops.
    - solver_time_limit: Max. Laufzeit für HiGHS (Sekunden).
    - epsilon: Puffer zwischen Operationen auf derselben Maschine.
    - arrival_column: Name der Spalte in df_arrivals für Ankunftszeiten.
    - reschedule_start: Zeitpunkt, ab dem neu geplant wird.

    Rückgabe:
    - df_schedule: DataFrame mit ['Job','Operation','Arrival','Machine','Start','Processing Time','Flow time','End']
    """
    # Vorbereitung: Ankunftszeiten & Sortierung
    df_arrivals = df_arrivals.sort_values(arrival_column).reset_index(drop=True)
    arrival_times = df_arrivals.set_index('Job')[arrival_column].to_dict()
    job_list = df_arrivals['Job'].tolist()
    num_jobs = len(job_list)

    # Job-Operationen aus df_jssp aufbereiten
    ops_grouped = df_jssp.sort_values(['Job', 'Operation']).groupby('Job')
    all_ops = []  # Liste pro Job: [(op_id, machine_id, duration), ...]
    all_machines = set()
    for job in job_list:
        grp = ops_grouped.get_group(job)
        ops = []
        for _, row in grp.iterrows():
            op_id = row['Operation']
            mac_str = str(row['Machine'])
            match = re.search(r"M(\d+)", mac_str)
            m_id = int(match.group(1)) if match else 0
            dur = float(row['Processing Time'])
            ops.append((op_id, m_id, dur))
            all_machines.add(m_id)
        all_ops.append(ops)

    # Letzte Endzeit bereits geplanter Operationen pro Job
    last_executed_end = df_executed.groupby('Job')['End'].max().to_dict()

    # Filter fixierte Ops ab reschedule_start
    df_executed_fixed = df_executed[df_executed['End'] >= reschedule_start].copy()
    df_executed_fixed['MachineID'] = (
        df_executed_fixed['Machine'].astype(str)
        .str.extract(r"M(\d+)", expand=False)
        .astype(int)
    )
    fixed_ops = {
        m: list(gr[['Start', 'End', 'Job']].itertuples(index=False, name=None))
        for m, gr in df_executed_fixed.groupby('MachineID')
    }

    # LP-Modell
    prob = pulp.LpProblem('JSSP_IndFlow_Fixed', pulp.LpMinimize)

    # Variablen
    starts = {
        (j, o): pulp.LpVariable(f"start_{j}_{o}", lowBound=0)
        for j in range(num_jobs)
        for o in range(len(all_ops[j]))
    }
    job_ends = {
        j: pulp.LpVariable(f"job_end_{j}", lowBound=0)
        for j in range(num_jobs)
    }

    # Zielfunktion: Summe (End - Arrival)
    prob += pulp.lpSum(
        (job_ends[j] - arrival_times[job_list[j]])
        for j in range(num_jobs)
    )

    # Technologische Reihenfolge + individuelle Startbedingungen
    for j, job in enumerate(job_list):
        earliest = max(
            arrival_times[job],
            last_executed_end.get(job, reschedule_start)
        )
        prob += starts[(j, 0)] >= earliest
        for o in range(1, len(all_ops[j])):
            _, _, prev_dur = all_ops[j][o-1]
            prob += starts[(j, o)] >= starts[(j, o-1)] + prev_dur

    # Maschinenkonflikte
    M = 1e5
    for m in sorted(all_machines):
        ops_on_m = [
            (j, o, all_ops[j][o][2])
            for j in range(num_jobs)
            for o, (_, mach, _) in enumerate(all_ops[j])
            if mach == m
        ]
        # Konflikte neu-neu
        for i in range(len(ops_on_m)):
            j1, o1, d1 = ops_on_m[i]
            for j2, o2, d2 in ops_on_m[i+1:]:
                if j1 == j2:
                    continue
                y = pulp.LpVariable(f"y_{j1}_{o1}_{j2}_{o2}", cat='Binary')
                prob += (starts[(j1, o1)] + d1 + epsilon
                         <= starts[(j2, o2)] + M*(1-y))
                prob += (starts[(j2, o2)] + d2 + epsilon
                         <= starts[(j1, o1)] + M*y)
        # Konflikte neu-fixiert
        for j1, o1, d1 in ops_on_m:
            for fixed_start, fixed_end, fixed_job in fixed_ops.get(m, []):
                y_fix = pulp.LpVariable(f"y_fix_{j1}_{o1}_{fixed_job}", cat='Binary')
                prob += (starts[(j1, o1)] + d1 + epsilon
                         <= fixed_start + M*(1-y_fix))
                prob += (fixed_end + epsilon
                         <= starts[(j1, o1)] + M*y_fix)

    # Endzeitbindung je Job
    for j in range(num_jobs):
        last_o = len(all_ops[j]) - 1
        prob += job_ends[j] >= starts[(j, last_o)] + all_ops[j][last_o][2]

    # Lösen
    prob.solve(pulp.HiGHS_CMD(msg=True, timeLimit=solver_time_limit))

    # Ergebnis extrahieren
    recs = []
    for (j, o), var in sorted(starts.items()):
        st = var.varValue
        if st is None:
            continue
        op_id, mach, dur = all_ops[j][o]
        end = st + dur
        recs.append({
            'Job': job_list[j],
            'Operation': op_id,
            'Arrival': arrival_times[job_list[j]],
            'Machine': f"M{mach}",
            'Start': round(st, 2),
            'Processing Time': dur,
            'Flow time': round(end - arrival_times[job_list[j]], 2),
            'End': round(end, 2)
        })

    df_schedule = pd.DataFrame(recs)
    # Spaltenreihenfolge sicherstellen
    cols = ['Job', 'Operation', 'Arrival', 'Machine', 'Start', 'Processing Time', 'Flow time', 'End']
    df_schedule = df_schedule[cols]
    df_schedule = df_schedule.sort_values(['Arrival', 'Start']).reset_index(drop=True)
    print("✅ Fertig!")
    return df_schedule

