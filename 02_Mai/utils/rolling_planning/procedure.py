import pandas as pd


# I) Init Filtern nach Teitfenster -------------------------------------------------------------------
def filter_jobs_by_arrival_window(
    df_times: pd.DataFrame,
    df_jssp: pd.DataFrame,
    day_start: float,
    planning_end: float,
    arrival_column: str = "Arrival"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filtert Jobs anhand eines Zeitfensters (z.B. Tagesabschnitt) und gibt die passenden
    Datensätze für Ankunftszeiten und JSSP zurück.

    Parameter:
    - df_times: DataFrame mit ['Job', arrival_column], z.B. Ankunftszeiten.
    - df_jssp: DataFrame mit ['Job', 'Operation', 'Machine', 'Processing Time'].
    - day_start: Startzeit des Zeitfensters.
    - planning_end: Endzeit des Zeitfensters.
    - arrival_column: Name der Spalte mit den Ankunftszeiten (Standard: 'Arrival').

    Rückgabe:
    - df_times_filtered: Nur Jobs, deren Arrival im Fenster liegt.
    - df_jssp_filtered: Entsprechende Operationen aus df_jssp.
    """
    time_filter = (df_times[arrival_column] >= day_start) & (df_times[arrival_column] < planning_end)
    df_times_filtered = df_times[time_filter].copy()
    relevant_jobs = df_times_filtered["Job"].unique()
    df_jssp_filtered = df_jssp[df_jssp["Job"].isin(relevant_jobs)].copy()
    return df_jssp_filtered, df_times_filtered

# II Änderungen --------------------------------------------------------------------------------------
# IIa eventuelle "Executed" Operations entfernen
def get_unexecuted_operations(
    df_jssp_filtered: pd.DataFrame,
    df_execution: pd.DataFrame
) -> pd.DataFrame:
    """
    Gibt alle Operationen aus df_jssp_filtered zurück, die noch nicht in df_execution enthalten sind.
    Nutzt einen Anti-Join auf ['Job', 'Operation'].

    Parameter:
    - df_jssp_filtered: DataFrame mit geplanten Operationen ['Job', 'Operation', ...].
    - df_execution: DataFrame mit ausgeführten Operationen ['Job', 'Operation', ...].

    Rückgabe:
    - df_jssp_todo: DataFrame mit noch auszuführenden Operationen.
    """
    jssp_keys = df_jssp_filtered[['Job', 'Operation']]
    execution_keys = df_execution[['Job', 'Operation']]

    merged = df_jssp_filtered.merge(
        execution_keys.drop_duplicates(),
        on=['Job', 'Operation'],
        how='left',
        indicator=True
    )

    df_jssp_todo = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])
    return df_jssp_todo

# IIb nicht angefangene Operations anhängen, die ursprünglich schon angefangen sein sollten

def extend_with_undone_operations(
    df_jssp_todo: pd.DataFrame,
    df_undone: pd.DataFrame
) -> pd.DataFrame:
    """
    Kombiniert noch nicht gestartete Operationen mit abgebrochenen Operationen.
    Achtet auf Einheitlichkeit der Datentypen und entfernt Duplikate.

    Parameter:
    - df_jssp_todo: DataFrame mit geplanten, aber noch nicht ausgeführten Operationen.
    - df_undone: DataFrame mit während des Tages abgebrochenen Operationen.

    Rückgabe:
    - df_jssp_todo_extended: Kombinierter, bereinigter DataFrame.
    """
    # Relevante Spalten aus df_undone
    df_undone_relevant = df_undone[['Job', 'Operation', 'Machine', 'Processing Time']].copy()

    # Einheitlicher Datentyp für 'Job'
    df_undone_relevant['Job'] = df_undone_relevant['Job'].astype(str)
    df_jssp_todo['Job'] = df_jssp_todo['Job'].astype(str)

    # Kombination beider DataFrames
    df_combined = pd.concat([df_undone_relevant, df_jssp_todo], ignore_index=True)

    # Doppelte Operationen entfernen
    df_combined.drop_duplicates(subset=['Job', 'Operation'], inplace=True)

    # Index neu setzen
    df_combined.reset_index(drop=True, inplace=True)

    return df_combined


# IIc  
def update_times_after_operation_changes(
    df_times: pd.DataFrame,
    df_jssp_todo_extended: pd.DataFrame,
    job_column: str = "Job"
) -> pd.DataFrame:
    """
    Aktualisiert df_times basierend auf dem aktuellen Stand von df_jssp_todo_extended.
    Entfernt veraltete Zeiteinträge und ergänzt ggf. fehlende, indem nur Jobs
    berücksichtigt werden, die tatsächlich noch geplante Operationen haben.

    Dies ist notwendig, wenn im Planungsprozess Operationen entfernt oder hinzugefügt wurden.

    Parameter:
    - df_times: Ursprünglicher DataFrame mit Zeiteinträgen (z.B. Ankunftszeiten).
    - df_jssp_todo_extended: Aktueller Satz geplanter Operationen.
    - job_column: Name der Spalte mit der Job-ID (Standard: 'Job').

    Rückgabe:
    - df_times_updated: Bereinigter df_times mit nur noch relevanten Jobs.
    """
    relevant_jobs = df_jssp_todo_extended[job_column].unique()
    df_times_updated = df_times[df_times[job_column].isin(relevant_jobs)].copy()
    return df_times_updated.reset_index(drop=True)

### III

def get_operations_running_into_day(df_execution: pd.DataFrame, day_start: float) -> pd.DataFrame:
    """
    Gibt alle Operationen zurück, deren Endzeit in oder nach dem gegebenen Tagesstart liegt.
    D.h. alle Operationen, die noch aktiv sind oder über den Tageswechsel hinauslaufen.

    Parameter:
    - df_execution: DataFrame mit mindestens der Spalte 'End'.
    - day_start: Startzeit des betrachteten Tages (z.B. 1440.0 für Tag 2 bei Minutenmodellierung).

    Rückgabe:
    - DataFrame mit relevanten Operationen.
    """
    return df_execution[df_execution["End"] >= day_start].copy()