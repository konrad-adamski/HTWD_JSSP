
import re
import math
import random
import time

import simpy
import pandas as pd
import pulp

from Machine import Machine

# --- Hilfsfunktionen ---

def get_time_str(minutes_in):
    minutes_total = int(minutes_in)
    seconds = int((minutes_in - minutes_total) * 60)
    hours = minutes_total // 60
    minutes = minutes_total % 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def get_duration(minutes_in):
    minutes = int(minutes_in)
    seconds = int(round((minutes_in - minutes) * 60))
    parts = []
    if minutes:
        parts.append(f"{minutes:02} minute{'s' if minutes != 1 else ''}")
    if seconds:
        parts.append(f"{seconds:02} second{'s' if seconds != 1 else ''}")
    return " ".join(parts) if parts else ""

def duration_log_normal(duration, vc=0.2):
    sigma = vc
    mu = math.log(duration)
    result = random.lognormvariate(mu, sigma)
    return round(result, 2)

def get_undone_operations_df(df_plan, df_exec):
    # Identifiziere Operations, die im Plan aber nicht in der Ausführung sind
    df_diff = pd.merge(
        df_plan[["Job", "Operation", "Machine"]],
        df_exec[["Job", "Operation", "Machine"]],
        how='outer',
        indicator=True
    ).query('_merge == "left_only"')[["Job", "Operation", "Machine"]]

    df_result = df_plan[["Job", "Operation", "Arrival", "Machine", "Start", "Processing Time"]].merge(
        df_diff,
        on=["Job", "Operation", "Machine"],
        how="inner"
    )
    df_result = df_result.rename(columns={"Start": "Planned Start"}).reset_index(drop=True)
    return df_result.sort_values(by="Planned Start")

# --- Scheduling-Funktion mit fixierten Operationen ---

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

    # LP-Modell aufsetzen
    prob = pulp.LpProblem('JSSP_IndFlow_Fixed', pulp.LpMinimize)

    # Variablen
    starts = {
        (j, o): pulp.LpVariable(f"start_{j}_{o}", lowBound=0)
        for j in range(num_jobs)
        for o in range(len(all_ops[j]))
    }
    job_ends = { j: pulp.LpVariable(f"job_end_{j}", lowBound=0) for j in range(num_jobs) }

    # Zielfunktion: Summe (End - Arrival)
    prob += pulp.lpSum(
        (job_ends[j] - arrival_times[job_list[j]])
        for j in range(num_jobs)
    )

    # Reihenfolge und individuelle Earliest-Start
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
                prob += (starts[(j1, o1)] + d1 + epsilon <= starts[(j2, o2)] + M*(1-y))
                prob += (starts[(j2, o2)] + d2 + epsilon <= starts[(j1, o1)] + M*y)
        # Konflikte neu-fixiert
        for j1, o1, d1 in ops_on_m:
            for fixed_start, fixed_end, fixed_job in fixed_ops.get(m, []):
                y_fix = pulp.LpVariable(f"y_fix_{j1}_{o1}_{fixed_job}", cat='Binary')
                prob += (starts[(j1, o1)] + d1 + epsilon <= fixed_start + M*(1-y_fix))
                prob += (fixed_end + epsilon <= starts[(j1, o1)] + M*y_fix)

    # Job-Ende definieren
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
    df_schedule = df_schedule[cols].sort_values(['Arrival', 'Start']).reset_index(drop=True)
    print("✅ Fertig!")
    return df_schedule

# --- Simulationsklasse ---

class ProductionDaySimulation:
    def __init__(self, dframe_schedule_plan, vc=0.2):
        self.start_time = 0
        self.end_time = 1440
        self.controller = None
        self.dframe_schedule_plan = dframe_schedule_plan
        self.vc = vc
        self.starting_times_dict = {}
        self.finished_log = []
        self.env = None
        self.stop_event = None
        self.machines = None

    def _init_machines(self):
        unique_machines = self.dframe_schedule_plan["Machine"].unique()
        return {m: Machine(self.env, m) for m in unique_machines}

    def job_process(self, job_id, job_operations):
        for op in job_operations:
            machine = self.machines[op["Machine"]]
            planned_start = op["Start"]
            planned_duration = op["Processing Time"]
            sim_duration = duration_log_normal(planned_duration, vc=self.vc)

            delay = max(planned_start - self.env.now, 0)
            yield self.env.timeout(delay)

            with machine.request() as req:
                yield req
                sim_start = self.env.now

                if self.job_cannot_start_on_time(job_id, machine, sim_start):
                    return

                self.job_started_on_machine(sim_start, job_id, machine)
                self.starting_times_dict[(job_id, machine.name)] = round(sim_start, 2)

                yield self.env.timeout(sim_duration)
                sim_end = self.env.now
                self.job_finished_on_machine(sim_end, job_id, machine, sim_duration)

            # Log-Eintrag inkl. Operation
            self.finished_log.append({
                "Job": job_id,
                "Operation": op["Operation"],
                "Machine": machine.name,
                "Start": round(sim_start, 2),
                "Simulated Processing Time": sim_duration,
                "End": round(sim_end, 2)
            })

            self.starting_times_dict.pop((job_id, machine.name), None)

            if self.env.now > self.end_time and not self.starting_times_dict:
                print(f"\n[{get_time_str(self.env.now)}] Simulation ended! There are no more active Operations")
                self.stop_event.succeed()

    def run(self, start_time=0, end_time=1440):
        self.start_time = start_time
        self.end_time = end_time
        self.env = simpy.Environment(initial_time=start_time)
        self.stop_event = self.env.event()
        self.machines = self._init_machines()

        for job_id, group in self.dframe_schedule_plan.groupby("Job"):
            operations = group.sort_values("Start").to_dict("records")
            self.env.process(self.job_process(job_id, operations))

        self.env.run(until=self.stop_event)

        dframe_execution = pd.DataFrame(self.finished_log)
        arrival_map = self.dframe_schedule_plan[["Job", "Operation", "Machine", "Arrival"]].drop_duplicates()
        dframe_execution = dframe_execution.merge(arrival_map, on=["Job","Operation","Machine"], how="left")
        dframe_execution["Flow time"] = dframe_execution["End"] - dframe_execution["Arrival"]
        dframe_execution = dframe_execution[["Job","Operation","Arrival","Machine","Start","Simulated Processing Time","Flow time","End"]]
        dframe_execution = dframe_execution.sort_values(["Arrival","Start","Job"]).reset_index(drop=True)

        dframe_undone = get_undone_operations_df(self.dframe_schedule_plan, dframe_execution)
        return dframe_execution, dframe_undone

    def job_started_on_machine(self, time_stamp, job_id, machine):
        print(f"[{get_time_str(time_stamp)}] {job_id} started on {machine.name}")
        if self.controller:
            self.controller.job_started_on_machine(time_stamp, job_id, machine)
            time.sleep(0.05)

    def job_finished_on_machine(self, time_stamp, job_id, machine, sim_duration):
        print(f"[{get_time_str(time_stamp)}] {job_id} finished on {machine.name} (after {get_duration(sim_duration)})")
        if self.controller:
            self.controller.job_finished_on_machine(time_stamp, job_id, machine, sim_duration)
            time.sleep(0.14)

    def job_cannot_start_on_time(self, job_id, machine, time_stamp):
        if time_stamp > self.end_time:
            print(
                f"[{get_time_str(time_stamp)}] {job_id} interrupted before machine "
                f"{machine.name} — would start too late (after {get_time_str(self.end_time)})"
            )
            return True
        return False

    def set_controller(self, controller):
        self.controller = controller
        self.controller.add_machines(*self.machines.values())
        job_ids = sorted(self.dframe_schedule_plan["Job"].unique())
        self.controller.update_jobs(*job_ids)

# --- Utils: JSSP-Dict ---

def get_jssp_from_schedule(df_schedule: pd.DataFrame, duration_column: str = "Processing Time") -> dict:
    job_dict = {}
    df_schedule = df_schedule.copy()
    df_schedule["Machine"] = df_schedule["Machine"].str.extract(r"M(\d+)").astype(int)
    df_schedule[duration_column] = df_schedule[duration_column].astype(int)

    for job, op_id, machine, duration in zip(
        df_schedule["Job"],
        df_schedule["Operation"],
        df_schedule["Machine"],
        df_schedule[duration_column]
    ):
        if job not in job_dict:
            job_dict[job] = []
        job_dict[job].append([machine, duration, op_id])

    return job_dict

# --- Main ---

if __name__ == "__main__":
    df_jssp = pd.read_csv("data/04_schedule_plan_firstday.csv")
    # Beispiel-Aufruf der Rescheduling-Funktion:
    # df_arrivals und df_executed müssten vorher geladen bzw. berechnet werden.
    # df_schedule = solve_jssp_individual_flowtime_with_fixed_ops(df_jssp, df_arrivals, df_executed)

    # Simulation mit dem Plan
    df_schedule_plan = pd.read_csv("data/04_schedule_plan_firstday.csv")
    simulation = ProductionDaySimulation(df_schedule_plan, vc=0.25)
    df_execution, df_undone = simulation.run(end_time=1440)

    print("\n=================== Abgeschlossene Operationen ===================")
    print(df_execution)
    print("\n======================= Offene Operationen =======================")
    print(df_undone)
    print("\n======================= JSSP-Dictionary =======================")
    for j, val in get_jssp_from_schedule(df_undone).items():
        print(f"{j}: {val}")
