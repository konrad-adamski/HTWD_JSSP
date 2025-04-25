import pandas as pd

def check_machine_conflicts(df_schedule: pd.DataFrame) -> pd.DataFrame | None:
    df = df_schedule.sort_values(["Machine", "Start"])  
    conflict_indices = []

    for machine in df["Machine"].unique():
        machine_df = df[df["Machine"] == machine].sort_values("Start")

        for i in range(1, len(machine_df)):
            prev = machine_df.iloc[i - 1]
            curr = machine_df.iloc[i]

            if curr["Start"] < prev["End"]:
                conflict_indices.extend([prev.name, curr.name])

    conflict_indices = sorted(set(conflict_indices))

    if conflict_indices:
        print(f"Maschinenkonflikte gefunden: {len(conflict_indices)} Zeilen betroffen.")
        return df_schedule.loc[conflict_indices].sort_values(["Machine", "Start"])
    else:
        print("Keine Maschinenkonflikte gefunden.")
        return None


def check_job_machine_sequence(df: pd.DataFrame, matrix: list) -> bool:
    violations = 0
    for job_id, job_ops in enumerate(matrix):
        job_df = df[df["Job"] == f"Job {job_id}"].copy()
        job_df.sort_values("Start", inplace=True)
        actual_sequence = job_df["Machine"].str.extract(r"M(\d+)").astype(int)[0].tolist()
        expected_sequence = [op[0] for op in job_ops]
        if actual_sequence != expected_sequence:
            print(f"  Reihenfolge-Verletzung bei Job {job_id}:")
            print(f"  Erwartet: {expected_sequence}")
            print(f"  Gefunden: {actual_sequence}")
            violations += 1
    print(f"\nAnzahl verletzter Job-Maschinen-Reihenfolgen: {violations}")
    return violations == 0


def check_job_machine_sequence_dict(df: pd.DataFrame, job_dict: dict) -> bool:
    violations = 0
    for job_name, job_ops in job_dict.items():
        job_df = df[df["Job"] == job_name].copy()
        job_df.sort_values("Start", inplace=True)
        actual_sequence = job_df["Machine"].str.extract(r"M(\d+)").astype(int)[0].tolist()
        expected_sequence = [op[0] for op in job_ops]
        if actual_sequence != expected_sequence:
            print(f"  Reihenfolge-Verletzung bei {job_name}:")
            print(f"  Erwartet: {expected_sequence}")
            print(f"  Gefunden: {actual_sequence}")
            violations += 1
    print(f"\nAnzahl verletzter Job-Maschinen-Reihenfolgen: {violations}")
    return violations == 0


def check_correct_start(df_schedule, df_arrivals):
    # Arrival-Dictionary bauen
    arrival_dict = dict(zip(df_arrivals["Job-ID"], df_arrivals["Ankunftszeit (Minuten)"]))

    # Ankunftszeit mappen
    df = df_schedule.copy()
    df["Ankunftszeit"] = df["Job"].map(arrival_dict)

    # Regelverletzungen: Start < Ankunft
    violations = df[df["Start"] < df["Ankunftszeit"]]

    if violations.empty:
        print("\nAlle Starts erfolgen nach der Ankunftszeit.")
        return None
    else:
        print("\nFehlerhafte Starts gefunden:")
        return violations
