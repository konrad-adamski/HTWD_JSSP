from scipy.stats import kendalltau
import pandas as pd

def analyze_machine_sequences(prev_plan: pd.DataFrame,
                              revised_plan: pd.DataFrame,
                              T1: float, with_position_change=False) -> pd.DataFrame:
    """
    Führt vollständige Analyse der Maschinenreihenfolgen durch:
    - Vergleicht Sequenzen nach T1
    - Ermittelt Job-Positionsänderungen
    - Berechnet Kendall's Tau
    """
    # Schritt 1: Sequenzen vergleichen
    df_compare = compare_machine_sequences_after_T1(prev_plan, revised_plan, T1)


    # Positionsänderungen ermitteln
    if with_position_change:
        df_compare["order_violations"] = df_compare.apply(
            lambda row: compute_job_position_changes(row["original_sequence"], row["revised_sequence"]),
            axis=1
        )
    

    # Kendall's Tau berechnen
    df_compare["kendall_tau"] = df_compare.apply(
        lambda row: kendall_tau_from_sequences(row["original_sequence"], row["revised_sequence"]),
        axis=1
    )

    return df_compare




def compare_machine_sequences_after_T1(prev_plan: pd.DataFrame,
                                       revised_plan: pd.DataFrame,
                                       T1: float) -> pd.DataFrame:
    """
    Vergleicht die Job-Reihenfolge je Maschine nach T1.

    Spalten:
    - original_sequence      : Reihenfolge der Jobs im Ursprungsplan (nach T1)
    - revised_sequence_raw   : Reihenfolge im Rescheduling-Plan (nach T1, ungefiltert)
    - revised_sequence       : revised_sequence_raw gefiltert auf original_sequence
    - changed                : True, wenn Reihenfolge geändert wurde
    """
    # Einheitliche Spalten
    df_prev = prev_plan.rename(columns={'Start': 'Start_prev', 'Job': 'Job'})
    df_rev  = revised_plan.rename(columns={'Start': 'Start_rev', 'ob': 'Job'})

    # Filter nach T1
    df_prev = df_prev[df_prev['Start_prev'] >= T1]
    df_rev  = df_rev[df_rev['Start_rev'] >= T1]

    # Sortieren
    df_prev_sel = df_prev[['Machine', 'Start_prev', 'Job']].sort_values(['Machine', 'Start_prev'])
    df_rev_sel  = df_rev[['Machine', 'Start_rev', 'Job']].sort_values(['Machine', 'Start_rev'])

    # Gruppieren
    original_sequence = df_prev_sel.groupby('Machine')['Job'].apply(list)
    revised_sequence_raw = df_rev_sel.groupby('Machine')['Job'].apply(list)

    # Vergleichstabelle
    comparison = pd.DataFrame({
        'original_sequence': original_sequence,
        'revised_sequence_raw': revised_sequence_raw
    }).dropna()

    # Gefilterte revised_sequence
    revised_filtered = {
        machine: [job for job in revised_sequence_raw.get(machine, []) if job in original_sequence[machine]]
        for machine in comparison.index
    }

    comparison['revised_sequence'] = pd.Series(revised_filtered)
    comparison['changed'] = comparison['original_sequence'] != comparison['revised_sequence']

    return comparison



def kendall_tau_from_sequences(original_sequence: list, revised_sequence: list) -> float:
    rank_original = {job: i for i, job in enumerate(original_sequence)}
    rank_revised = [rank_original[job] for job in revised_sequence]
    baseline = list(range(len(rank_revised)))
    tau, _ = kendalltau(baseline, rank_revised)
    return tau


def compute_job_position_changes(original_sequence: list, revised_sequence: list) -> list:
    """
    Gibt Liste von Jobs zurück, deren Position sich geändert hat
    (inkl. der alten und neuen Position).
    """
    # Index-Tabellen
    original_pos = {job: i for i, job in enumerate(original_sequence)}
    revised_pos  = {job: i for i, job in enumerate(revised_sequence)}

    # Vergleich je Job – nur wenn in beiden vorhanden
    changed = []
    for job in original_sequence:
        if job in revised_pos:
            old = original_pos[job]
            new = revised_pos[job]
            if old != new:
                changed.append((job, old, new))  # (Jobname, Originalpos, Neuepos)

    return changed

