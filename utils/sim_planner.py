import pandas as pd
import utils.editor as edit
import utils.ProductionDaySimulation as daysim
from utils.checker import check_all_constraints
from utils.scheduler import generate_job_arrivals
from utils.scheduler import solve_jobshop_optimal
from utils.scheduler import solve_stage2_early_starts

def print_jobs_compact(job_dict):
    for job, tasks in job_dict.items():
        print(f" {job}:\t{tasks}")
    print("")



def plan_day(matrix_index, this_day_jobs, remaining_jobs, solver_time_limit=300, working_day_hours=24):
    print(f"\n==== Tag {matrix_index} ====")

    # Schritt 1: neue Ankunftszeiten erzeugen
    df_arrivals = generate_job_arrivals(this_day_jobs, u_b_mmax=0.9, day_id=matrix_index)

    # Falls noch verbliebene Job-Operationen vom Vortag da sind, hinzufügen
    if remaining_jobs is not None:
        df_arrivals = edit.add_remaining_jobs_with_zero_arrival(df_arrivals, remaining_jobs, day_id=matrix_index)
        this_day_jobs = edit.merge_jobs(remaining_jobs, this_day_jobs)
        print("Folgende Jobs vom Vortag werden berücksichtigt:")
        print_jobs_compact(remaining_jobs)
    

    # Schritt 2: Erster Jobshop-Solver (Stage 1)
    df_schedule_highs, opt_makespan = solve_jobshop_optimal(
        this_day_jobs, df_arrivals, day_id=matrix_index, solver_time_limit=solver_time_limit
    )

    # Schritt 3: Zweiter Solver für frühe Starts (Stage 2)
    df_schedule_early_starts, final_makespan = solve_stage2_early_starts(
        this_day_jobs, df_arrivals, opt_makespan, day_id=matrix_index, solver_time_limit=solver_time_limit
    )
    print(f"\tFinaler Makespan: {final_makespan} Minuten")

    # Überprüfen
    check_all_constraints(df_schedule_early_starts, this_day_jobs, df_arrivals)

    # Schritt 4: Zeitplan aufteilen
    df_schedule_on_time, df_late = edit.separate_operation_by_day_limit(df_schedule_early_starts, day_limit_h=working_day_hours)

    # Schritt 5: Übrig gebliebene Jobs extrahieren
    remaining_jobs = edit.get_jssp_from_schedule(df_late) if not df_late.empty else None

    if remaining_jobs is not None:
        print(f"\tAnzahl verbliebener (geplanter) Jobs für nächsten Tag: {len(remaining_jobs)}")
    else:
        print("\tNach Plan sollen alle Jobs rechtzeitig abgeschlossen werden, ohne Überträge auf den nächsten Tag.")

    return remaining_jobs, df_schedule_on_time


def plan_days_in_horizon(current_day, planning_horizon, job_set_list, remaining_jobs, remaining_jobs_current_day,
                         max_solver_time, working_day_hours=24):
    df_schedule_daily = pd.DataFrame()

    for day_id in range(current_day, current_day + planning_horizon):
        if day_id >= len(job_set_list):
            print(f" Tag {day_id} überschreitet die verfügbare Instanzliste. Abbruch.")
            break

        this_day_jobs = job_set_list[day_id]

        remaining_jobs, df_schedule_planned_on_time = plan_day(
            day_id, this_day_jobs, remaining_jobs, max_solver_time, working_day_hours
        )
        df_schedule_daily = pd.concat([df_schedule_daily, df_schedule_planned_on_time], ignore_index=True)

        if day_id == current_day:
            remaining_jobs_current_day = remaining_jobs

    # --- Simulation NACH der Planung ---
    df_schedule_current_day = df_schedule_daily[df_schedule_daily["Day-ID"] == current_day]

    if not df_schedule_current_day.empty:
        print(f"\n----- Simulation für Simulationstag {current_day} -----\n")
        simulation = daysim.ProductionDaySimulation(df_schedule_current_day, vc=0.25)
        df_execution, df_undone = simulation.run(until=1440)
        sim_undone_jobs = edit.get_jssp_from_schedule(df_undone, duration_column="Planned Duration")
        print(" Folgende Jobs wurden bei der Simulation nicht abgeschlossen:")
        print_jobs_compact(sim_undone_jobs)
        remaining_jobs_current_day = edit.merge_jobs(remaining_jobs_current_day, sim_undone_jobs)

    return df_schedule_daily, remaining_jobs_current_day


def run_simulation(df_schedule_daily_list, job_set_list, current_day, remaining_jobs_current_day, total_simulation_days,
                   max_planning_horizon, max_solver_time, working_day_hours=24):
    while current_day < total_simulation_days:
        remaining_jobs = remaining_jobs_current_day
        print(f"\n########## Planung ab Simulationstag {current_day} ##########")

        planning_horizon = min(max_planning_horizon, total_simulation_days - current_day)
        print(f" Planungshorizont: {planning_horizon} Tage (von Tag {current_day} bis {current_day + planning_horizon - 1})")

        df_schedule_daily, remaining_jobs_current_day = plan_days_in_horizon(
            current_day, planning_horizon, job_set_list, remaining_jobs,
            remaining_jobs_current_day, max_solver_time, working_day_hours
        )

        df_schedule_daily_list.append(df_schedule_daily)

        current_day += 1

    return df_schedule_daily_list