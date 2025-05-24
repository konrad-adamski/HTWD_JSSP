import utils.rolling_planning.init_jobs_times as rp_init
import utils.rolling_planning.procedure as rp_proced
import utils.reschedule.schedule_solver__tardiness_plus as rssv_t

import utils.basics.presenter as show
import utils.checker as check
from ProductionDaySimulation import ProductionDaySimulation


def get_schedule_filename(day: int, suffix: str = "", prefix: str = "07") -> str:
    file_template = "data/{prefix}_schedule_{day:02d}{suffix}.csv"
    if suffix:
        suffix = f"_{suffix}"
    return file_template.format(prefix=prefix,day=day, suffix=suffix)


def run_multi_day_rescheduling(first_start, last_planning_start, day_length, horizon_days, 
                               df_times, df_jssp, 
                               df_execution, df_undone, df_plan, notebook_prefix="XX", rescheduler= "bi_criteria_sum_tardiness_deviation", solver_limit=7200,
                               plot_results=True, vc=0.35, this_r=0.4):
    """
    Führt eine mehrtägige Rescheduling-Simulation durch.
    """

    for day_numb in range(first_start, last_planning_start + 1):
        day_start = day_length * day_numb
        day_end = day_start + day_length
        planning_end = day_start + horizon_days * day_length

        # ------------------- I. Ankunfts- und Operationsvorbereitung -------------------
        df_jssp_curr, df_times_curr = rp_proced.filter_jobs_by_arrival_window(df_times, df_jssp, day_start, planning_end)
        df_jssp_curr = rp_proced.extend_with_undone_operations(df_jssp_curr, df_undone)
        df_times_curr = rp_proced.update_times_after_operation_changes(df_times, df_jssp_curr)

        # ------------------- II. Neue (zukünftige) Jobs hinzufügen -------------------
        df_jssp_curr, df_times_curr = rp_init.add_beforehand_jobs_to_current_horizon(
            df_existing_jobs=df_jssp_curr,
            df_existing_times=df_times_curr,
            df_jssp=df_jssp,
            df_times=df_times,
            min_arrival_time=planning_end,
            n=3,
            random_state=23
        )

        # ------------------- III. Relevante laufende Operationen -------------------
        df_execution_important = rp_proced.get_operations_running_into_day(df_execution, day_start)

        # ------------------- IV. Rescheduling durchführen -------------------

        if rescheduler  == "bi_criteria_sum_tardiness_deviation": 
            print("bi_criteria_sum_tardiness_deviation ...")
            df_plan = rssv_t.solve_jssp_bi_criteria_sum_tardiness_deviation_with_fixed_ops(
                df_jssp=df_jssp_curr,
                df_arrivals_deadlines=df_times_curr,
                df_executed=df_execution_important,
                df_original_plan=df_plan,
                r=this_r,
                solver_time_limit=solver_limit,
                reschedule_start=day_start,
                threads=8
            )
        else:
            print("sum_tardiness_with_fixed_ops ...")
            df_plan = rssv_t.solve_jssp_sum_tardiness_with_fixed_ops(
                df_jssp=df_jssp_curr,
                df_arrivals_deadlines=df_times_curr,
                df_executed=df_execution_important,
                solver_time_limit=solver_limit,
                reschedule_start=day_start,
                threads=8
            )

        df_plan.to_csv(get_schedule_filename(day=day_numb, prefix=notebook_prefix), index=False)
        
        if plot_results:
            show.plot_gantt_machines(df_plan, title=f"Gantt-Diagramm ab Tag {day_numb}")
        check.check_all_constraints(df_plan)

        # ------------------- V. Einen Tag simulieren -------------------
        simulation = ProductionDaySimulation(df_plan, vc=vc)
        df_execution, df_undone = simulation.run(start_time=day_start, end_time=day_end)
        if not df_execution.empty:
            if plot_results:
                show.plot_gantt_machines(df_execution, title=f"Gantt-Diagramm für Simulationstag {day_numb}", duration_column="Simulated Processing Time")
        else:
            print(f"Nothing executed on day {day_numb}")

