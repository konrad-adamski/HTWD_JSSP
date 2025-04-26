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


# V2 -----------------------------------------------------------------------------------------------


def is_machine_conflict_free(df_schedule: pd.DataFrame) -> bool:
    """
    Prüft, ob es Maschinenkonflikte gibt.
    Gibt True zurück, wenn konfliktfrei.
    Gibt False zurück und druckt die Konflikte, wenn Konflikte existieren.
    """
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


def is_start_correct_(df_schedule: pd.DataFrame, df_arrivals: pd.DataFrame) -> bool:
    """
    Prüft, ob alle Jobs frühestens ab ihrer Ankunftszeit starten.
    Gibt True zurück, wenn alle Starts korrekt sind.
    Gibt False zurück und zeigt fehlerhafte Starts, wenn vorhanden.
    """
    # Arrival-Dictionary bauen
    arrival_dict = dict(zip(df_arrivals["Job-ID"], df_arrivals["Ankunftszeit (Minuten)"]))

    # Ankunftszeit mappen
    df = df_schedule.copy()
    df["Ankunftszeit"] = df["Job"].map(arrival_dict)

    # Regelverletzungen: Start < Ankunft
    violations = df[df["Start"] < df["Ankunftszeit"]]

    if violations.empty:
        return True
    else:
        print(f"Fehlerhafte Starts gefunden ({len(violations)} Zeilen):")
        print(violations.sort_values("Start"))
        return False



def check_all_constraints(df_schedule: pd.DataFrame, job_dict: dict, df_arrivals: pd.DataFrame) -> bool:
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

    if not is_start_correct_(df_schedule, df_arrivals):
        checks_passed = False

    if checks_passed:
        print("✅ Alle Constraints wurden erfüllt.\n")
    else:
        print("❗ Es wurden Constraint-Verletzungen gefunden.\n")

    return checks_passed


