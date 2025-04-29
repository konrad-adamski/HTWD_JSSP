import math
import random
import simpy
import pandas as pd
import time

def get_time_str(minutes_in):
    """Wandle Minuten in HH:MM:SS um, einfach basierend auf Minuten."""
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
    """
    Liefert eine Lognormal-verteilte Zufallsdauer basierend auf:
    - duration: gewünschter Mittelwert
    - vc: Variationskoeffizient
    Achtung: Erwartungswert liegt oberhalb von duration.
    """
    sigma = vc
    mu = math.log(duration)
    result = random.lognormvariate(mu, sigma)
    return round(result, 2)

def get_undone_operations_df(df_plan, df_exec):
    """
    Bestimmt alle Operationen aus df_plan, deren (Job, Machine)-Paar
    nicht in df_exec vorhanden ist, und benennt 'Duration' zu 'Planned Duration' um.
    """
    df_diff = pd.merge(
        df_plan[["Job", "Machine"]],
        df_exec[["Job", "Machine"]],
        how='outer',
        indicator=True
    ).query('_merge == "left_only"').drop(columns=['_merge'])

    df_result = df_plan[["Job", "Machine", "Duration"]].merge(
        df_diff,
        on=["Job", "Machine"],
        how="inner"
    )

    return df_result.rename(columns={"Duration": "Planned Duration"})


# --- Klasse für Maschinen ---

class Machine(simpy.Resource):
    def __init__(self, env, name):
        super().__init__(env, capacity=1)
        self.name = name


# --- Hauptklasse: Tagesproduktion simulieren ---
class ProductionDaySimulation:
    def __init__(self, dframe_schedule_plan, vc=0.2):
        """
        Args:
            dframe_schedule_plan (DataFrame): Geplanter Tagesplan mit Spalten wie 'Job', 'Machine', 'Start', 'Duration', 'End'
            vc (float): Variationskoeffizient für Lognormalverteilung
        """
        self.controller = None
        self.until = 1440

        self.dframe_schedule_plan = dframe_schedule_plan
        self.vc = vc
        self.env = simpy.Environment()
        #self.env = simpy.rt.RealtimeEnvironment(factor=1/12)  # 1/12 -> langsam; 1/18 -> mittel; 1/22 -> schnell
        self.machines = self._init_machines()

        self.starting_times_dict= {}
        self.finished_log = []

    def _init_machines(self):
        unique_machines = self.dframe_schedule_plan["Machine"].unique()
        return {m: Machine(self.env, m) for m in unique_machines}

    def job_process(self, job_id, job_operations):
        for op in job_operations:
            machine = self.machines[op["Machine"]]
            planned_start = op["Start"]
            planned_duration = op["Duration"]

            sim_duration = duration_log_normal(planned_duration, vc=self.vc)

            # Warten bis zum geplanten Start (wenn nötig)
            delay = max(planned_start - self.env.now, 0)
            yield self.env.timeout(delay)

            with machine.request() as req:
                yield req
                sim_start = self.env.now

                if self.job_cannot_finish_on_time(job_id, machine, sim_start, planned_duration):
                    return  # GANZEN JOB abbrechen

                self.job_started_on_machine(sim_start, job_id, machine)
                self.starting_times_dict[(job_id, machine.name)] = round(sim_start, 2)

                yield self.env.timeout(sim_duration)
                sim_end = self.env.now
                self.job_finished_on_machine(sim_end, job_id, machine, sim_duration)

            self.finished_log.append({ "Job": job_id, "Machine": machine.name, "Start": round(sim_start, 2),
                                        "Duration": sim_duration, "End": round(sim_end, 2)
                                        })
            # Fertige Operationen werden aus der starting_times Dictionary entfernt
            if (job_id, machine.name) in self.starting_times_dict:
                del self.starting_times_dict[(job_id, machine.name)]

    def run(self, until=None):
        """
        Startet die Simpy-Simulation und gibt zwei DataFrames zurück:
        - df_execution: tatsächlich ausgeführte Operationen
        - df_late: Operationen, die nicht ausgeführt wurden
        """
        if until is not None:
            self.until = min(until, 1440)
        jobs_grouped = self.dframe_schedule_plan.groupby("Job")


        for job_id, group in jobs_grouped:
            # --- Sortiere Operationen nach geplanter Startzeit ---
            operations = group.sort_values("Start").to_dict("records")
            self.env.process(self.job_process(job_id, operations))

        self.env.run(until=self.until)

        dframe_execution = pd.DataFrame(self.finished_log)
        dframe_undone = get_undone_operations_df(self.dframe_schedule_plan, dframe_execution)

        # Hinzufügen des Starts (falls gestartet, aber nicht fertig geworden)
        dframe_undone["Start"] = dframe_undone.apply(
            lambda row: self.starting_times_dict.get((row["Job"], row["Machine"])),
            axis=1
        )
        return dframe_execution, dframe_undone

    # Ausgaben ------------------------------------------------------------------
    def job_started_on_machine(self, time_stamp, job_id, machine):
        print(f"[{get_time_str(time_stamp)}] {job_id} started on {machine.name}")
        if self.controller:
            self.controller.job_started_on_machine(time_stamp, job_id, machine)
            time.sleep(0.04)

    def job_finished_on_machine(self, time_stamp, job_id, machine, sim_duration):
        print(f"[{get_time_str(time_stamp)}] {job_id} finished on {machine.name} (after {get_duration(sim_duration)})")
        if self.controller:
            self.controller.job_finished_on_machine(time_stamp, job_id, machine, sim_duration)
            time.sleep(0.12)

    def job_cannot_finish_on_time(self, job_id, machine, time_stamp, planned_duration):
        """
        Prüft, ob eine Operation zu spät starten würde,
        sodass sie nicht mehr innerhalb des Tages abgeschlossen werden kann.
        """
        if time_stamp + planned_duration > self.until:
            print(
                f"[{get_time_str(time_stamp)}] {job_id} interrupted before machine "
                f"{machine.name} — would finish too late at {get_time_str(time_stamp + planned_duration)}"
            )
            return True
        return False

    # Controller
    def set_controller(self, controller):
        self.controller = controller
        self.controller.add_machines(*self.machines.values())

        job_ids = job_ids = sorted(self.dframe_schedule_plan["Job"].unique())
        self.controller.update_jobs(*job_ids)


