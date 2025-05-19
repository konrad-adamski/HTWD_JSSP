import math
import random
import time

import simpy
import pandas as pd

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

         # Nach der letzen Operation des Jobs: Warten bis Mindestzeit erreicht
        if self.env.now < self.end_time:
            remaining = self.end_time - self.env.now
            yield self.env.timeout(remaining)
    
        # Wenn jetzt keine Jobs mehr laufen → Simulation sauber beenden
        if not self.starting_times_dict and not self.stop_event.triggered:
            print(f"\n[{get_time_str(self.env.now)}]  Simulation ended! There are no active Operations.")
            self.stop_event.succeed()
                    
    def run(self, start_time=0, end_time=1440):
        self.start_time = start_time
        self.end_time = end_time

        self.env = simpy.Environment(initial_time=start_time)
        self.stop_event = self.env.event()
        self.machines = self._init_machines()
    
        if self.dframe_schedule_plan.empty:
            print("⚠️ Kein Zeitplan vorhanden – Simulation wird übersprungen.")
            return pd.DataFrame(), pd.DataFrame()
    
        jobs_started = 0
        for job_id, group in self.dframe_schedule_plan.groupby("Job"):
            operations = group.sort_values("Start").to_dict("records")
            if operations and operations[0]["Start"] <= self.end_time:
                self.env.process(self.job_process(job_id, operations))
                jobs_started += 1
    
        if jobs_started == 0:
            print("⚠️ Kein Job startet innerhalb des Zeitfensters – Simulation wird abgebrochen.")
            return pd.DataFrame(), pd.DataFrame()
    
        try:
            self.env.run(until=self.stop_event)
        except RuntimeError as e:
            print("⚠️ Simulation wurde unerwartet beendet:", e)
            return pd.DataFrame(), pd.DataFrame()

        
        dframe_execution = pd.DataFrame(self.finished_log)
        # Arrival aus df_plan mappen
        arrival_map = self.dframe_schedule_plan[["Job", "Machine", "Arrival"]].drop_duplicates()
        dframe_execution = dframe_execution.merge(arrival_map, on=["Job", "Machine"], how="left")

        # Flow time berechnen
        dframe_execution["Flow time"] = dframe_execution["End"] - dframe_execution["Arrival"]

        dframe_execution = dframe_execution[[
            "Job", "Operation", "Arrival", "Machine", "Start", "Simulated Processing Time", "Flow time", "End"
        ]].sort_values(by=["Arrival", "Start", "Job"]).reset_index(drop=True)

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

