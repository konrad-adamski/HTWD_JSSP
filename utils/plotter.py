import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_gantt_jobs(schedule_df: pd.DataFrame, title: str = "Gantt-Diagramm"):
    # Farbzuordnung für Maschinen
    machines = sorted(schedule_df['Machine'].unique())
    color_map = {machine: plt.cm.tab20(i % 20) for i, machine in enumerate(machines)}

    # Gantt-Diagramm: Y-Achse = Jobs, Farben = Maschinen
    fig, ax = plt.subplots(figsize=(14, 8))
    jobs = sorted(schedule_df['Job'].unique())
    yticks = range(len(jobs))

    for idx, job in enumerate(jobs):
        job_ops = schedule_df[schedule_df['Job'] == job]
        for _, row in job_ops.iterrows():
            color = color_map[row['Machine']]
            ax.barh(idx, row['Duration'], left=row['Start'], height=0.5, color=color)

    # Legende (Maschine → Farbe)
    legend_handles = [mpatches.Patch(color=color_map[m], label=f"{m}") for m in machines]
    ax.legend(handles=legend_handles, title="Maschinen", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Achsen, Titel
    ax.set_yticks(yticks)
    ax.set_yticklabels(jobs)
    ax.set_xlabel("Zeit")
    ax.set_ylabel("Jobs")
    ax.set_title(title)
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def plot_gantt_machines(schedule_df: pd.DataFrame, title: str = "Gantt-Diagramm (Maschinenansicht)"):
    # Farbzuordnung für Jobs
    jobs = sorted(schedule_df['Job'].unique())
    color_map = {job: plt.cm.tab10(i % 10) for i, job in enumerate(jobs)}

    # Gantt-Diagramm: Y-Achse = Maschinen, Farben = Jobs
    fig, ax = plt.subplots(figsize=(14, 8))
    machines = sorted(schedule_df['Machine'].unique())
    yticks = range(len(machines))

    for idx, machine in enumerate(machines):
        ops = schedule_df[schedule_df['Machine'] == machine]
        for _, row in ops.iterrows():
            color = color_map[row['Job']]
            ax.barh(idx, row['Duration'], left=row['Start'], height=0.5, color=color)

    # Legende (Job → Farbe)
    legend_handles = [mpatches.Patch(color=color_map[job], label=job) for job in jobs]
    ax.legend(handles=legend_handles, title="Jobs", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Achsen & Formatierung
    ax.set_yticks(yticks)
    ax.set_yticklabels(machines)
    ax.set_xlabel("Zeit")
    ax.set_ylabel("Maschinen")
    ax.set_title(title)
    ax.grid(True)
    plt.tight_layout()
    plt.show()


