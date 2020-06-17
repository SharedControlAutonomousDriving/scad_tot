import sys, re

# ^INFO:sensitivity:(x\d+_s\d+)\s([\w\s]+):\s(?:(\([\d.,\s)]+)|[\d.]+\s)\((\d+ms)\)$
sensitivity_regex, robustness_regex = r'^INFO:sensitivity:', r'^INFO:robustness:'
sensitivity_log_regex = r'^INFO:sensitivity:(x\d+_s\d+)\s([\w\s]+):\s(?:(\([\d.,\s)]+)|[\d.]+\s)\((\d+)ms\)$'
robustness_log_regex = r'^INFO:robustness:(s\d+)\s([\w\s]+):\s(?:(\([\d.,\s)]+)|[\d.]+\s)\((\d+)ms\)$'

def get_durations(logfile):
    durations, logtype = [], None
    with open(logfile, 'r') as f:
        # read first line to find out what type of log file it is.
        logtype = 'sensitivity' if re.match(sensitivity_regex, f.readline().strip()) else 'robustness'
        duration_regex = sensitivity_log_regex if logtype == 'sensitivity' else robustness_log_regex
        for l in f:
            matches = re.search(duration_regex, l.strip(), re.MULTILINE)
            if matches:
                durations.append((matches[1], matches[2], int(matches[4])))
    
    aggregated_durations = {}
    for d in durations:
        identifier, _, duration = d
        current = aggregated_durations[identifier] if aggregated_durations.get(identifier) else 0
        aggregated_durations[identifier] = current + duration
    
    if logtype == 'sensitivity':
        sample_durations = {}
        for aggid,d in aggregated_durations.items():
            xid, sid = aggid.split('_')
            if not sample_durations.get(sid):
                sample_durations[sid] = {'total': 0}
            sample_durations[sid][xid] = d
            sample_durations[sid]['total'] = sample_durations[sid]['total'] + d
        aggregated_durations = sample_durations
        total = sum([s['total'] for s in aggregated_durations.values()])
        sample_average = total / len(aggregated_durations)
        features_average = sum([d for s in aggregated_durations.values() for d in s.values()]) / sum([len(s) for s in aggregated_durations.values()])
        aggregated_durations['features_average'] = features_average
    else:
        total = sum([s for s in aggregated_durations.values()])
        sample_average = sum([s for s in aggregated_durations.values()]) / len(aggregated_durations)
    aggregated_durations['sample_average'] = sample_average
    aggregated_durations['totaltime'] = total
    return aggregated_durations

if __name__ == '__main__':
    fname = sys.argv[1]
    d = get_durations(fname)
    
    print(f'Durations from {fname}:')
    print(d)
