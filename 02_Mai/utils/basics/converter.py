import pandas as pd


def jssp_dict_to_df(jobs_dict: dict) -> pd.DataFrame:
    """
    Wandelt ein Dictionary mit Job-Operationen in einen DataFrame um.

    Parameter:
    - jobs_dict: dict
        Schlüssel sind Job-Bezeichnungen (z.B. 'job 0'),
        Werte sind Listen von [Machine, Processing Time].

    Rückgabe:
    - pd.DataFrame mit Spalten ['Job', 'Operation', 'Machine', 'Processing Time'].
      Die Spalte 'Operation' enthält die Reihenfolge der Operation innerhalb des Jobs.
    """
    records = []
    for job, ops in jobs_dict.items():
        for op_idx, (machine_idx, proc_time) in enumerate(ops):
            records.append({
                'Job': job,
                'Operation': op_idx,
                'Machine': f'M{machine_idx}',
                'Processing Time': proc_time
            })
    # DataFrame mit definierter Spaltenreihenfolge erzeugen
    df = pd.DataFrame(records, columns=['Job', 'Operation', 'Machine', 'Processing Time'])
    return df
