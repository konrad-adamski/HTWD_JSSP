import pandas as pd
import numpy as np
import random
import pulp

# Mittlere Zwischenankuftszeit (single day) -----------------------------------------

def get_engpassmaschine_from_dict(job_dict):
    machine_usage = {}
    for job_ops in job_dict.values():
        for machine, duration in job_ops:
            machine_usage[machine] = machine_usage.get(machine, 0) + duration
    return max(machine_usage, key=machine_usage.get)



def get_vec_t_b_mmax_from_dict(job_dict):
    # Engpassmaschine bestimmen
    engpassmaschine = get_engpassmaschine_from_dict(job_dict)

    # Vektor der Bearbeitungszeiten auf der Engpassmaschine
    vec_t_b_mmax = []
    for job in job_dict.values():
        duration = next((d for m, d in job if m == engpassmaschine), 0)
        vec_t_b_mmax.append(duration)

    return vec_t_b_mmax


def get_interarrival_time_from_dict(job_dict, u_b_mmax=0.9):
    n_jobs = len(job_dict)
    p = [1 / n_jobs] * n_jobs  # Gleichverteilung
    vec_t_b_mmax = get_vec_t_b_mmax_from_dict(job_dict)  # Engpass-Zeiten je Job

    # Erwartungswert der Zeit auf der Engpassmaschine, skaliert durch Auslastung
    t_a = sum(p[i] * vec_t_b_mmax[i] for i in range(n_jobs)) / u_b_mmax
    return np.round(t_a, 2)


# Ankunftszeiten ---------------------------------------------------------------------

def generate_job_arrivals_df(job_dict, u_b_mmax=0.9, day_id=0, random_seed_jobs=12, random_seed_times=123):
    job_names = list(job_dict.keys())  # z. B. ['job 00_0', ..., 'job 00_9']
    n_jobs = len(job_names)

    # Permutiere Jobnamen
    np.random.seed(random_seed_jobs)
    shuffled_jobs = list(np.random.permutation(job_names))

    # Interarrival-Zeit auf Basis der Engpassmaschine
    t_a = get_interarrival_time_from_dict(job_dict, u_b_mmax=u_b_mmax)

    # Erzeuge Ankunftszeiten
    np.random.seed(random_seed_times)
    interarrival_times = np.random.exponential(scale=t_a, size=n_jobs)
    arrival_times = np.round(np.cumsum(interarrival_times), 2)

    df_day = pd.DataFrame({
        "Job-ID": shuffled_jobs,
        "Day-ID": [day_id] * n_jobs,
        "Ankunftszeit (Minuten)": arrival_times
    })

    return df_day


def add_day(existing_df, job_dict, u_b_mmax=0.9, day_id=None):
    if day_id is None:
        day_id = 0 if existing_df.empty else existing_df["Day-ID"].max() + 1

    df_new_day = generate_jobs_for_single_day(job_dict, u_b_mmax, day_id=day_id)

    return pd.concat([existing_df, df_new_day], ignore_index=True)



# HiGHs --------------------------------------------------------


def solve_jobshop_optimal(job_dict, df_arrivals, day_id=0, solver_time_limit=600, epsilon=0.06):
    """
    Erste Stufe: Minimierung des Makespan (Gesamtdauer) eines Job-Shop-Problems.

    Parameter:
    - epsilon: Kleiner Sicherheitsabstand (in Minuten) zwischen Operationen auf derselben Maschine,
               um numerische Ungenauigkeiten und Maschinenkonflikte zu vermeiden (z.B. 0.06 Minuten = 3.6 Sekunden).
    """

    job_names = list(job_dict.keys())
    num_jobs = len(job_names)
    all_ops = list(job_dict.values())

    # Maschinen extrahieren
    all_machines = {op[0] for job in all_ops for op in job}

    # LP-Problem definieren
    prob = pulp.LpProblem("JobShop_Optimal_HiGHS", pulp.LpMinimize)

    # Variablen: Startzeiten
    starts = {
        (j, o): pulp.LpVariable(f"start_{j}_{o}", lowBound=0, cat="Continuous")
        for j in range(num_jobs) for o in range(len(all_ops[j]))
    }

    # Makespan-Variable
    makespan = pulp.LpVariable("makespan", lowBound=0, cat="Continuous")
    prob += makespan  # Ziel 1: Makespan minimieren

    # Ankunftszeiten berücksichtigen
    arrival_times = df_arrivals.set_index("Job-ID")["Ankunftszeit (Minuten)"].to_dict()

    # Technologische Reihenfolge und Ankunftszeit
    for j, job_name in enumerate(job_names):
        job = job_dict[job_name]
        prob += starts[(j, 0)] >= arrival_times[job_name]
        for o in range(1, len(job)):
            d_prev = job[o - 1][1]
            prob += starts[(j, o)] >= starts[(j, o - 1)] + d_prev

    # Maschinenkonflikte (mit kleinem Abstand epsilon)
    bigM = 1e5
    for m in all_machines:
        ops = [(j, o, d) for j in range(num_jobs)
               for o, (mach, d) in enumerate(all_ops[j]) if mach == m]
        for i in range(len(ops)):
            j1, o1, d1 = ops[i]
            for j2, o2, d2 in ops[i + 1:]:
                if j1 != j2:
                    y = pulp.LpVariable(f"y_{j1}_{o1}_{j2}_{o2}", cat="Binary")
                    prob += starts[(j1, o1)] + d1 + epsilon <= starts[(j2, o2)] + bigM * (1 - y)
                    prob += starts[(j2, o2)] + d2 + epsilon <= starts[(j1, o1)] + bigM * y

    # Makespan-Bedingung für jede Job-Endoperation
    for j in range(num_jobs):
        last_op = len(all_ops[j]) - 1
        prob += makespan >= starts[(j, last_op)] + all_ops[j][last_op][1]

    # Solver starten
    solver = pulp.HiGHS_CMD(msg=True, timeLimit=solver_time_limit)
    prob.solve(solver)

    # Ergebnisse extrahieren
    schedule_data = []
    for (j, o), var in sorted(starts.items()):
        start = var.varValue
        if start is not None:
            machine, duration = all_ops[j][o]
            end = start + duration
            schedule_data.append({
                "Job": job_names[j],
                "Machine": f"M{machine}",
                "Day-ID": day_id,
                "Start": round(start, 2),
                "Duration": duration,
                "End": round(end, 2)
            })

    df_schedule = pd.DataFrame(schedule_data)
    makespan_value = round(pulp.value(makespan), 3)

    return df_schedule, makespan_value

def solve_stage2_early_starts(job_dict, df_arrivals, optimal_makespan, day_id=0, solver_time_limit=300, epsilon=0.06):
    """
    Zweite Stufe: Minimierung der Summe der Startzeiten unter Fixierung des Makespan (lexikographische Optimierung).

    Parameter:
    - epsilon: Kleiner Sicherheitsabstand (in Minuten) zwischen Operationen auf derselben Maschine,
               um numerische Ungenauigkeiten und Maschinenkonflikte zu vermeiden (z.B. 0.06 Minuten = 3.6 Sekunden).
    """

    job_names = list(job_dict.keys())
    num_jobs = len(job_names)
    all_ops = list(job_dict.values())
    all_machines = {op[0] for job in all_ops for op in job}

    prob = pulp.LpProblem("JobShop_Secondary_EarlyStart", pulp.LpMinimize)

    # Startzeit-Variablen
    starts = {
        (j, o): pulp.LpVariable(f"start_{j}_{o}", lowBound=0, cat="Continuous")
        for j in range(num_jobs) for o in range(len(all_ops[j]))
    }

    arrival_times = df_arrivals.set_index("Job-ID")["Ankunftszeit (Minuten)"].to_dict()

    # Technologische Reihenfolge und Ankunftszeiten
    for j, job_name in enumerate(job_names):
        job = job_dict[job_name]
        prob += starts[(j, 0)] >= arrival_times[job_name]
        for o in range(1, len(job)):
            d_prev = job[o - 1][1]
            prob += starts[(j, o)] >= starts[(j, o - 1)] + d_prev

    # Maschinenkonflikte (mit epsilon)
    bigM = 1e5
    for m in all_machines:
        ops = [(j, o, d) for j in range(num_jobs)
               for o, (mach, d) in enumerate(all_ops[j]) if mach == m]
        for i in range(len(ops)):
            j1, o1, d1 = ops[i]
            for j2, o2, d2 in ops[i + 1:]:
                if j1 != j2:
                    y = pulp.LpVariable(f"y_{j1}_{o1}_{j2}_{o2}", cat="Binary")
                    prob += starts[(j1, o1)] + d1 + epsilon <= starts[(j2, o2)] + bigM * (1 - y)
                    prob += starts[(j2, o2)] + d2 + epsilon <= starts[(j1, o1)] + bigM * y

    # Fixierter Makespan: Endzeit der Jobs darf den optimalen Makespan nicht überschreiten
    for j in range(num_jobs):
        last_op = len(all_ops[j]) - 1
        prob += starts[(j, last_op)] + all_ops[j][last_op][1] <= optimal_makespan

    # Ziel 2: Minimierung der Summe der Startzeiten (frühe Starts bevorzugen)
    total_start = pulp.lpSum([starts[(j, 0)] for j in range(num_jobs)])
    prob += total_start

    # Solver starten
    solver = pulp.HiGHS_CMD(msg=True, timeLimit=solver_time_limit)
    prob.solve(solver)

    # Ergebnisse extrahieren
    schedule_data = []
    for (j, o), var in sorted(starts.items()):
        start = var.varValue
        if start is not None:
            machine, duration = all_ops[j][o]
            end = start + duration
            schedule_data.append({
                "Job": job_names[j],
                "Machine": f"M{machine}",
                "Day-ID": day_id,
                "Start": round(start, 2),
                "Duration": duration,
                "End": round(end, 2)
            })

    df_schedule = pd.DataFrame(schedule_data)
    return df_schedule, optimal_makespan




# v2
def solve_stage2_early_starts_all(job_dict, df_arrivals, optimal_makespan, day_id=0, solver_time_limit=300, epsilon=0.06):
    """
    Zweite Stufe: Minimierung der Summe aller Startzeiten unter Fixierung des Makespan (verbesserte Version).

    Parameter:
    - job_dict: Dictionary der Jobs mit Maschinen- und Bearbeitungszeiten.
    - df_arrivals: DataFrame mit Ankunftszeiten der Jobs.
    - optimal_makespan: Optimales Makespan aus Stufe 1 (fester Endzeitpunkt).
    - day_id: Tag-Identifikator für Mehrtagesplanung.
    - solver_time_limit: Zeitlimit für den Solver (Sekunden).
    - epsilon: Kleiner Sicherheitsabstand zwischen Operationen auf derselben Maschine.
    """

    job_names = list(job_dict.keys())
    num_jobs = len(job_names)
    all_ops = list(job_dict.values())
    all_machines = {op[0] for job in all_ops for op in job}

    prob = pulp.LpProblem("JobShop_Secondary_EarlyStart", pulp.LpMinimize)

    # Startzeit-Variablen
    starts = {
        (j, o): pulp.LpVariable(f"start_{j}_{o}", lowBound=0, cat="Continuous")
        for j in range(num_jobs) for o in range(len(all_ops[j]))
    }

    # Ankunftszeiten vorbereiten
    arrival_times = df_arrivals.set_index("Job-ID")["Ankunftszeit (Minuten)"].to_dict()

    # Technologische Reihenfolge und Ankunftszeiten einfügen
    for j, job_name in enumerate(job_names):
        job = job_dict[job_name]
        prob += starts[(j, 0)] >= arrival_times[job_name]
        for o in range(1, len(job)):
            d_prev = job[o - 1][1]
            prob += starts[(j, o)] >= starts[(j, o - 1)] + d_prev

    # Maschinenkonflikte (mit epsilon Abstand)
    bigM = 1e5
    for m in all_machines:
        ops = [(j, o, d) for j in range(num_jobs)
               for o, (mach, d) in enumerate(all_ops[j]) if mach == m]
        for i in range(len(ops)):
            j1, o1, d1 = ops[i]
            for j2, o2, d2 in ops[i + 1:]:
                if j1 != j2:
                    y = pulp.LpVariable(f"y_{j1}_{o1}_{j2}_{o2}", cat="Binary")
                    prob += starts[(j1, o1)] + d1 + epsilon <= starts[(j2, o2)] + bigM * (1 - y)
                    prob += starts[(j2, o2)] + d2 + epsilon <= starts[(j1, o1)] + bigM * y

    # Fixierter Makespan: Endzeit der letzten Operationen darf den optimalen Makespan nicht überschreiten
    for j in range(num_jobs):
        last_op = len(all_ops[j]) - 1
        prob += starts[(j, last_op)] + all_ops[j][last_op][1] <= optimal_makespan

    # ZIEL: Minimierung der Summe aller Startzeiten (nicht nur der ersten Operationen)
    total_start = pulp.lpSum([starts[(j, o)] for j in range(num_jobs) for o in range(len(all_ops[j]))])
    prob += total_start

    # Solver starten
    solver = pulp.HiGHS_CMD(msg=True, timeLimit=solver_time_limit)
    prob.solve(solver)

    # Ergebnisse extrahieren
    schedule_data = []
    for (j, o), var in sorted(starts.items()):
        start = var.varValue
        if start is not None:
            machine, duration = all_ops[j][o]
            end = start + duration
            schedule_data.append({
                "Job": job_names[j],
                "Machine": f"M{machine}",
                "Day-ID": day_id,
                "Start": round(start, 2),
                "Duration": duration,
                "End": round(end, 2)
            })

    df_schedule = pd.DataFrame(schedule_data)
    return df_schedule, optimal_makespan








