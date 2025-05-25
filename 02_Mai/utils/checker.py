import pandas as pd


# V3 -----------------------------------------------------------------------------------------------


def is_machine_conflict_free(df_schedule: pd.DataFrame) -> bool:
    """
    Prüft, ob es Maschinenkonflikte gibt.
    Gibt True zurück, wenn konfliktfrei.
    Gibt False zurück und druckt die Konflikte, wenn Konflikte existieren.
    """
    df = df_schedule.sort_values(["Machine", "Start"]).reset_index()
    conflict_indices = []

    for machine in df["Machine"].unique():
        machine_df = df[df["Machine"] == machine].sort_values("Start")

        for i in range(1, len(machine_df)):
            prev = machine_df.iloc[i - 1]
            curr = machine_df.iloc[i]

            if curr["Start"] < prev["End"]:
                conflict_indices.extend([prev["index"], curr["index"]])

    conflict_indices = sorted(set(conflict_indices))

    if conflict_indices:
        print(f"Maschinenkonflikte gefunden: {len(conflict_indices)} Zeilen betroffen.")
        print(df_schedule.loc[conflict_indices].sort_values(["Machine", "Start"]))
        return False
    else:
        print("✅ Keine Maschinenkonflikte gefunden")
        return True


# neu
def is_operation_sequence_correct(df_schedule: pd.DataFrame) -> bool:
    """
    Prüft, ob für jeden Job die Operationen fortlaufend, aufsteigend und lückenlos sind.

    Parameter:
    - df_schedule: DataFrame mit Spalten ['Job', 'Operation', …]

    Rückgabe:
    - True, wenn jeder Job eine fortlaufende und lückenlose Operationssequenz aufweist.
      Sonst False und Ausgabe der betroffenen Jobs.
    """
    violations = []
    for job, grp in df_schedule.groupby('Job', sort=False):
        ops = sorted(grp['Operation'].tolist())
        expected = list(range(ops[0], ops[-1] + 1))
        if ops != expected:
            violations.append((job, expected, ops))

    if not violations:
        print("✅ Für alle Jobs ist die Operationssequenz korrekt.")
        return True

    print(f"❌ Fehler in der Operationssequenz bei {len(violations)} Job(s):")
    for job, exp, act in violations:
        print(f"  Job {job!r}: erwartet Indizes {exp}, gefunden {act}")
    return False




def is_start_correct(df_schedule: pd.DataFrame) -> bool:
    """
    Prüft, ob alle Operationen frühestens ab ihrer Ankunftszeit starten.
    Erwartet, dass 'Arrival' bereits in df_schedule vorhanden ist.
    """
    violations = df_schedule[df_schedule["Start"] < df_schedule["Arrival"]]

    if violations.empty:
        print("✅ Alle Operation starten erst nach Arrival des Job")
        return True
    else:
        print(f"Fehlerhafte Starts gefunden ({len(violations)} Zeilen):")
        print(violations.sort_values("Start"))
        return False


def is_job_timing_correct(df_schedule: pd.DataFrame) -> bool:
    """
    Prüft nur die zeitliche technologische Reihenfolge pro Job:
    Jede Operation muss frühestens nach Ende der vorherigen starten.
    Gibt True zurück, wenn alles korrekt ist.
    Gibt False zurück und zeigt fehlerhafte Zeitabfolgen.
    """
    df = df_schedule.copy()
    df = df.sort_values(["Job", "Start"]).reset_index(drop=True)

    violations = []

    for job in df["Job"].unique():
        df_job = df[df["Job"] == job].sort_values("Start").reset_index(drop=True)

        for i in range(1, len(df_job)):
            prev_end = df_job.loc[i - 1, "End"]
            curr_start = df_job.loc[i, "Start"]
            if curr_start < prev_end:
                violations.append((job, i, round(prev_end, 2), round(curr_start, 2)))

    if not violations:
        print("✅ Zeitliche technologische Reihenfolge korrekt.")
        return True
    else:
        print(f"❌ {len(violations)} Zeitverletzungen gefunden:")
        for job, i, prev_end, curr_start in violations:
            print(f"  ⏱️  Job {job} – Operation {i} startet zu früh: {curr_start} < {prev_end}")
        return False


def check_all_constraints(df_schedule: pd.DataFrame) -> bool:
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

    if not is_operation_sequence_correct(df_schedule):
        checks_passed = False

    if not is_start_correct(df_schedule):
        checks_passed = False

    if not is_job_timing_correct(df_schedule):
        checks_passed = False

    if checks_passed:
        print("\n✅ Alle Constraints wurden erfüllt.\n")
    else:
        print("\n❗ Es wurden Constraint-Verletzungen gefunden.\n")

    return checks_passed


