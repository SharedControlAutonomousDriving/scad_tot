import pandas as pd
# read the csv data
def crunch(path):
    details = pd.read_csv(path)

    # filter out zero epsilons
    details = details[details['leps'] > 0]
    # print(details)
    # get overall min epsilon
    overall = details['leps'].min()
    # print(overall)
    # get min epsilons for each label
    mineps = details.groupby('spred')['leps'].min()
    print(mineps)
    fast, med_fast, med, med_slow, slow = mineps
    
    print('\n'.join([
    f'overall: {overall}',
    f'----------------------',
    f'fast: {fast}',
    f'med-fast: {med_fast}',
    f'med: {med}',
    f'med-slow: {med_slow}',
    f'slow: {slow}',
    ]))
