import pandas as pd
import math


def estimate_bigM_with_deadline(df_jssp: pd.DataFrame, df_arrivals_deadlines: pd.DataFrame) -> float:
    """
    Schätzt einen realistischen BigM-Wert basierend auf den gegebenen JSSP-Daten.

    Parameter:
    - df_jssp: DataFrame mit ['Job','Operation','Machine','Processing Time']
    - df_arrivals_deadlines: DataFrame mit ['Job','Arrival','Deadline']

    Rückgabe:
    - Ein geeigneter BigM-Wert als float
    """
    max_deadline = df_arrivals_deadlines["Deadline"].max()
    max_arrival = df_arrivals_deadlines["Arrival"].max()
    max_processing = df_jssp["Processing Time"].max()
    
    # Worst-case Zeithorizont (alle Jobs seriell abgearbeitet)
    total_processing_time = df_jssp.groupby("Job")["Processing Time"].sum().sum()
    
    # Sicherheitsfaktor
    buffer = 0.1 * total_processing_time  # 10% Puffer

    # BigM: Maximum von mehreren konservativen Schätzern
    bigM = max(
        max_deadline + max_processing,
        total_processing_time + buffer,
        max_arrival + total_processing_time
    )
    
    return math.ceil(bigM*1.1 / 1000) * 1000 # bis zu 10% Puffer


def estimate_bigM_with_deadline_and_original_plan(
    df_jssp: pd.DataFrame,
    df_arrivals_deadlines: pd.DataFrame,
    df_original_plan: pd.DataFrame
) -> float:
    max_deadline = df_arrivals_deadlines["Deadline"].max()
    max_arrival = df_arrivals_deadlines["Arrival"].max()
    max_processing = df_jssp["Processing Time"].max()
    max_original_start = df_original_plan["Start"].max()
    min_original_start = df_original_plan["Start"].min()

    total_processing_time = df_jssp["Processing Time"].sum()
    buffer = 0.1 * total_processing_time

    bigM = max(
        max_deadline + max_processing,
        max_arrival + total_processing_time,
        max_original_start + max_processing,
        abs(min_original_start) + total_processing_time,
        total_processing_time + buffer
    )

    return math.ceil(bigM*1.1 / 1000) * 1000  # bis zu 10% Puffer
