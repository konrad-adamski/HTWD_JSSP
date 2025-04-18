import pandas as pd

def check_machine_conflicts(df):
    # Sortiere DataFrame nach Machine und Startzeit
    df_sorted = df.sort_values(by=["Machine", "Start"]).reset_index(drop=True)

    conflict_count = 0

    # Überprüfe jede Maschine einzeln
    for machine in df_sorted["Machine"].unique():
        machine_df = df_sorted[df_sorted["Machine"] == machine].sort_values(by="Start")

        # Zeilenweise durchgehen und Startzeiten vergleichen
        for i in range(1, len(machine_df)):
            prev_end = machine_df.iloc[i - 1]["End"]
            curr_start = machine_df.iloc[i]["Start"]

            if curr_start < prev_end:
                conflict_count += 1

    print(f"Gefundene Konflikte auf Maschinen: {conflict_count}")
    return conflict_count == 0


def check_job_machine_sequence(df, job_matrix):
    violations = 0

    for job_id, job_ops in enumerate(job_matrix):
        # Hole Einträge dieses Jobs aus dem DataFrame
        job_df = df[df["Job"] == f"Job {job_id}"].copy()
        job_df.sort_values("Start", inplace=True)

        # Extrahiere tatsächliche Maschinenreihenfolge aus dem Zeitplan
        actual_sequence = job_df["Machine"].str.extract(r"M(\d+)").astype(int)[0].tolist()
        expected_sequence = [op[0] for op in job_ops]

        if actual_sequence != expected_sequence:
            print(f"  Reihenfolge-Verletzung bei Job {job_id}:")
            print(f"  Erwartet: {expected_sequence}")
            print(f"  Gefunden: {actual_sequence}")
            violations += 1

    print(f"\nAnzahl verletzter Job-Maschinen-Reihenfolgen: {violations}")
    return violations == 0