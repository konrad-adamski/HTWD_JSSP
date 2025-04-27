import math
import random
import simpy
import pandas as pd

# --- Hilfsfunktionen ---

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

def cannot_finish_on_time(job_id, machine_name, sim_start, planned_duration, until):
    """
    Prüft, ob eine Operation zu spät starten würde,
    sodass sie nicht mehr innerhalb des Tages abgeschlossen werden kann.
    """
    if until is not None and sim_start + planned_duration > until:
        print(
            f"[{sim_start:.1f}] {job_id} interrupted before machine "
            f"{machine_name} — would finish too late at {sim_start + planned_duration:.1f}"
        )
        return True
    return False


class ProductionDaySimulation:
    def __init__(self, dframe_schedule_plan, vc=0.2):
        """
        Args:
            dframe_schedule_plan (DataFrame): Geplanter Tagesplan mit Spalten wie 'Job', 'Machine', 'Start', 'Duration', 'End'
            vc (float): Variationskoeffizient für Lognormalverteilung
        """
        self.dframe_schedule_plan = dframe_schedule_plan
        self.vc = vc
        self.env = simpy.Environment()
        self.machines = self._init_machines()

        self.starting_times_dict= {}
        self.finished_log = []

    def _init_machines(self):
        unique_machines = self.dframe_schedule_plan["Machine"].unique()
        return {m: Machine(self.env, m) for m in unique_machines}

    def job_process(self, job_id, job_operations, until=None):
        for op in job_operations:
            machine = self.machines[op["Machine"]]
            planned_start = op["Start"]
            planned_duration = op["Duration"]

            sim_duration = duration_log_normal(planned_duration, vc=self.vc)

            delay = planned_start - self.env.now
            if delay > 0:
                yield self.env.timeout(delay)

            with machine.request() as req:
                yield req
                sim_start = self.env.now

                if cannot_finish_on_time(job_id, machine.name, sim_start, planned_duration, until):
                    return  # GANZEN JOB abbrechen

                print(f"[{sim_start:.1f}] {job_id} started on {machine.name}")
                self.starting_times_dict[(job_id, machine.name)] = round(sim_start, 2)

                yield self.env.timeout(sim_duration)
                sim_end = self.env.now
                print(f"[{sim_end:.1f}] {job_id} finished on {machine.name} after {sim_duration} minutes")

            self.finished_log.append({ "Job": job_id, "Machine": machine.name, "Start": round(sim_start, 2),
                                        "Duration": sim_duration, "End": round(sim_end, 2)
                                        })
            if (job_id, machine.name) in self.starting_times_dict:
                del self.starting_times_dict[(job_id, machine.name)]

    def run(self, until=None):
        """
        Startet die Simpy-Simulation und gibt zwei DataFrames zurück:
        - df_execution: tatsächlich ausgeführte Operationen
        - df_late: Operationen, die nicht ausgeführt wurden
        """
        jobs_grouped = self.dframe_schedule_plan.groupby("Job")

        for job_id, group in jobs_grouped:
            # --- Sortiere Operationen nach geplanter Startzeit ---
            operations = group.sort_values("Start").to_dict("records")
            self.env.process(self.job_process(job_id, operations, until=until))

        if until is not None:
            self.env.run(until=until)
        else:
            self.env.run()

        dframe_execution = pd.DataFrame(self.finished_log)
        dframe_undone = get_undone_operations_df(self.dframe_schedule_plan, dframe_execution)

        # Hinzufügen des Starts (falls gestartet, aber nicht fertig geworden)
        dframe_undone["Start"] = dframe_undone.apply(
            lambda row: self.starting_times_dict.get((row["Job"], row["Machine"])),
            axis=1
        )


        return dframe_execution, dframe_undone


def get_jssp_from_schedule(df_schedule: pd.DataFrame, duration_column: str = "Duration") -> dict:
    job_dict = {}

    df_schedule = df_schedule.copy()
    df_schedule["Machine"] = df_schedule["Machine"].str.extract(r"M(\d+)").astype(int)
    df_schedule[duration_column] = df_schedule[duration_column].astype(int)

    for job, machine, duration in zip(df_schedule["Job"], df_schedule["Machine"], df_schedule[duration_column]):
        if job not in job_dict:
            job_dict[job] = []
        job_dict[job].append([machine, duration])

    return job_dict


if __name__ == "__main__":
    df_schedule_plan = pd.read_csv("data/schedule.csv")  # dein geplanter Tagesplan
    simulation = ProductionDaySimulation(df_schedule_plan, vc=0.25)
    df_execution, df_undone = simulation.run(until=1440)

    print("=== Abgeschlossene Operationen ===")
    print(df_execution)

    print("\n=== Offene (nicht abgeschlossene) Operationen ===")
    print(df_undone)

    print("\n====================================================")
    for j, val in get_jssp_from_schedule(df_undone, duration_column="Planned Duration").items():
        print(j + ": " + str(val))


