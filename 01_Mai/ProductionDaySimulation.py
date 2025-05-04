import math
import random
import time

import simpy
import pandas as pd

# from GUI.Controller import Controller
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
    df_diff = pd.merge(
        df_plan[["Job", "Machine"]],
        df_exec[["Job", "Machine"]],
        how='outer',
        indicator=True
    ).query('_merge == "left_only"').drop(columns=['_merge'])

    df_result = df_plan[["Job","Arrival", "Machine", "Start", "Processing Time"]].merge(
        df_diff,
        on=["Job", "Machine"],
        how="inner"
    )
    df_result = df_result.rename(columns={"Start": "Planned Start"}).reset_index(drop=True)

    return df_result.sort_values(by="Planned Start")


# --- Hauptklasse ---

class ProductionDaySimulation:
    def __init__(self, dframe_schedule_plan, vc=0.2):
        self.start_time = 0
        self.end_time = 1440  # 1 Tag

        self.controller = None

        self.dframe_schedule_plan = dframe_schedule_plan
        self.vc = vc

        self.starting_times_dict = {}
        self.finished_log = []

        self.env = None
        self.stop_event = None  # Abbruchsignal
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

            # Warten bis zum geplanten Start (wenn nötig)
            delay = max(planned_start - self.env.now, 0)
            yield self.env.timeout(delay)

            with machine.request() as req:
                yield req
                sim_start = self.env.now

                if self.job_cannot_start_on_time(job_id, machine, sim_start):
                    return  # GANZEN JOB abbrechen

                self.job_started_on_machine(sim_start, job_id, machine)
                self.starting_times_dict[(job_id, machine.name)] = round(sim_start, 2)

                yield self.env.timeout(sim_duration)
                sim_end = self.env.now
                self.job_finished_on_machine(sim_end, job_id, machine, sim_duration)

            self.finished_log.append({ "Job": job_id, "Machine": machine.name, "Start": round(sim_start, 2),
                                        "Simulated Processing Time": sim_duration, "End": round(sim_end, 2)
                                        })

            # Fertige Operationen werden aus der starting_times Dictionary entfernt
            if (job_id, machine.name) in self.starting_times_dict:
                del self.starting_times_dict[(job_id, machine.name)]

            # ✅ Abbruchbedingung: nach 1440 Minuten und nichts mehr aktiv
            if self.env.now > self.end_time and not self.starting_times_dict:
                print(f"\n[{get_time_str(self.env.now)}] Simulation ended! There are no more active Operations")
                self.stop_event.succeed()

    def run(self, start_time=0, end_time=1440):

        self.start_time = start_time
        self.end_time = end_time

        self.env = simpy.Environment(initial_time=start_time)
        self.stop_event = self.env.event()
        self.machines = self._init_machines()

        jobs_grouped = self.dframe_schedule_plan.groupby("Job")

        for job_id, group in jobs_grouped:
            operations = group.sort_values("Start").to_dict("records")
            self.env.process(self.job_process(job_id, operations))

        self.env.run(until=self.stop_event)  # Simulation läuft weiter bis alle gestarteten Jobs fertig sind

        dframe_execution = pd.DataFrame(self.finished_log)
        # Arrival aus df_plan mappen
        arrival_map = self.dframe_schedule_plan[["Job", "Machine", "Arrival"]].drop_duplicates()
        dframe_execution = dframe_execution.merge(arrival_map, on=["Job", "Machine"], how="left")

        # Flow time berechnen
        dframe_execution["Flow time"] = dframe_execution["End"] - dframe_execution["Arrival"]

        dframe_execution = dframe_execution[[
            "Job", "Arrival", "Machine", "Start", "Simulated Processing Time", "Flow time", "End"
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


# --- Utils ---

def get_jssp_from_schedule(df_schedule: pd.DataFrame, duration_column: str = "Processing Time") -> dict:
    job_dict = {}

    df_schedule = df_schedule.copy()
    df_schedule["Machine"] = df_schedule["Machine"].str.extract(r"M(\d+)").astype(int)
    df_schedule[duration_column] = df_schedule[duration_column].astype(int)

    for job, machine, duration in zip(df_schedule["Job"], df_schedule["Machine"], df_schedule[duration_column]):
        if job not in job_dict:
            job_dict[job] = []
        job_dict[job].append([machine, duration])

    return job_dict


# --- Main ---

if __name__ == "__main__":
    df_schedule_plan = pd.read_csv("data/04_schedule_plan_firstday.csv")  # Pfad zur neuen Datei

    simulation = ProductionDaySimulation(df_schedule_plan, vc=0.25)
    df_execution, df_undone = simulation.run(end_time=1440)

    print("\n=================== Abgeschlossene Operationen ===================")
    print(df_execution)

    print("\n======================= Offene Operationen =======================")
    print(df_undone)

    print("\n=======================")
    for j, val in get_jssp_from_schedule(df_undone).items():
        print(j + ": " + str(val))
