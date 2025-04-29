import pandas as pd

def schedule_spt(matrix):
    num_jobs = len(matrix)
    num_ops = len(matrix[0])

    job_ready = [0] * num_jobs
    machine_ready = {}
    job_ops_pointer = [0] * num_jobs
    schedule = []

    while any(p < num_ops for p in job_ops_pointer):
        available_ops = []

        for job_id in range(num_jobs):
            op_idx = job_ops_pointer[job_id]
            if op_idx < num_ops:
                machine, duration = matrix[job_id][op_idx]
                ready_time = max(job_ready[job_id], machine_ready.get(machine, 0))
                available_ops.append({
                    'JobID': job_id,
                    'Machine': machine,
                    'Duration': duration,
                    'ReadyTime': ready_time
                })

        selected = min(available_ops, key=lambda x: x['Duration'])

        job_id = selected['JobID']
        machine = selected['Machine']
        duration = selected['Duration']
        start = selected['ReadyTime']
        end = start + duration

        schedule.append({
            'Job': f'Job {job_id}',
            'Machine': f'M{machine}',
            'Start': start,
            'Duration': duration,
            'End': end
        })

        job_ready[job_id] = end
        machine_ready[machine] = end
        job_ops_pointer[job_id] += 1

    return pd.DataFrame(schedule)