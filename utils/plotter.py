import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_gantt_jobs(schedule_df: pd.DataFrame, title: str = "Gantt-Diagramm"):
    machines = sorted(schedule_df['Machine'].unique())
    
    # Bessere Farbskala: nipy_spectral für mehr Kontrast
    cmap = plt.cm.get_cmap("nipy_spectral", len(machines))
    color_map = {machine: cmap(i) for i, machine in enumerate(machines)}

    fig, ax = plt.subplots(figsize=(14, 8))
    jobs = sorted(schedule_df['Job'].unique())
    yticks = range(len(jobs))

    for idx, job in enumerate(jobs):
        job_ops = schedule_df[schedule_df['Job'] == job]
        for _, row in job_ops.iterrows():
            color = color_map[row['Machine']]
            ax.barh(idx, row['Duration'], left=row['Start'], height=0.5, color=color, edgecolor='black')

    # Legende
    legend_handles = [mpatches.Patch(color=color_map[m], label=f"{m}") for m in machines]
    ax.legend(handles=legend_handles, title="Maschinen", bbox_to_anchor=(1.05, 1), loc='upper left')

    ax.set_yticks(yticks)
    ax.set_yticklabels(jobs)
    ax.set_xlabel("Zeit")
    ax.set_ylabel("Jobs")
    ax.set_title(title)
    ax.grid(True)

    max_time = schedule_df['Start'] + schedule_df['Duration']
    ax.set_xlim(left=0, right=max(max_time) * 1.05)
    plt.tight_layout()
    plt.show()



def plot_gantt_machines(schedule_df: pd.DataFrame, title: str = "Gantt-Diagramm (Maschinenansicht)"):
    jobs = sorted(schedule_df['Job'].unique())

    # Farbskala mit hoher Unterscheidbarkeit (gleich wie bei plot_gantt_jobs)
    cmap = plt.cm.get_cmap("nipy_spectral", len(jobs))
    color_map = {job: cmap(i) for i, job in enumerate(jobs)}

    fig, ax = plt.subplots(figsize=(14, 8))
    machines = sorted(schedule_df['Machine'].unique())
    yticks = range(len(machines))

    for idx, machine in enumerate(machines):
        ops = schedule_df[schedule_df['Machine'] == machine]
        for _, row in ops.iterrows():
            color = color_map[row['Job']]
            ax.barh(idx, row['Duration'], left=row['Start'], height=0.5, color=color, edgecolor='black')

    # Legende (Jobs)
    legend_handles = [mpatches.Patch(color=color_map[job], label=job) for job in jobs]
    ax.legend(handles=legend_handles, title="Jobs", bbox_to_anchor=(1.05, 1), loc='upper left')

    ax.set_yticks(yticks)
    ax.set_yticklabels(machines)
    ax.set_xlabel("Zeit")
    ax.set_ylabel("Maschinen")
    ax.set_title(title)
    ax.grid(True)

    max_time = schedule_df['Start'] + schedule_df['Duration']
    ax.set_xlim(left=0, right=max(max_time) * 1.05)
    plt.tight_layout()
    plt.show()




