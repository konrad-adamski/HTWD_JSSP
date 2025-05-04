import pandas as pd


# V2 -----------------------------------------------------------------------------------------------


def is_machine_conflict_free(df_schedule: pd.DataFrame) -> bool:
    """
    Prüft, ob es Maschinenkonflikte gibt.
    Gibt True zurück, wenn konfliktfrei.
    Gibt False zurück und druckt die Konflikte, wenn Konflikte existieren.
    """
    df = df_schedule.sort_values(["Machine", "Start"]).reset_index(drop=True)
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
        print(df_schedule.loc[conflict_indices].sort_values(["Machine", "Start"]))
        return False
    else:
        return True

def is_job_machine_sequence_correct(df: pd.DataFrame, job_dict: dict) -> bool:
    violations = []

    for job, ops in job_dict.items():
        expected = [op[0] for op in ops]
        actual = (
            df[df["Job"] == job]
            .sort_values("Start")["Machine"]
            .str.extract(r"M(\d+)")
            .astype(int)[0]
            .tolist()
        )
        if actual != expected:
            violations.append((job, expected, actual))

    if not violations:
        return True

    print(f"Reihenfolge-Verletzungen bei {len(violations)} Jobs:")
    for job, expected, actual in violations:
        print(f"  Job {job}:\n    Erwartet: {expected}\n    Gefunden: {actual}")
    return False




def is_start_correct(df_schedule: pd.DataFrame) -> bool:
    """
    Prüft, ob alle Operationen frühestens ab ihrer Ankunftszeit starten.
    Erwartet, dass 'Arrival' bereits in df_schedule vorhanden ist.
    """
    violations = df_schedule[df_schedule["Start"] < df_schedule["Arrival"]]

    if violations.empty:
        return True
    else:
        print(f"Fehlerhafte Starts gefunden ({len(violations)} Zeilen):")
        print(violations.sort_values("Start"))
        return False




def check_all_constraints(df_schedule: pd.DataFrame, job_dict: dict) -> bool:
    """
    Führt alle wichtigen Prüfungen auf einem Tages-Schedule durch:
    - Maschinenkonflikte
    - Job-Maschinen-Reihenfolge
    - Startzeiten nach Ankunft
    Gibt True zurück, wenn alle Prüfungen bestanden sind, sonst False.
    """

    checks_passed = True

    if not is_machine_conflict_free(df_schedule):
        checks_passed = False

    if not is_job_machine_sequence_correct(df_schedule, job_dict):
        checks_passed = False

    if not is_start_correct(df_schedule):
        checks_passed = False

    if checks_passed:
        print("\t✅ Alle Constraints wurden erfüllt.\n")
    else:
        print("\t❗ Es wurden Constraint-Verletzungen gefunden.\n")

    return checks_passed


