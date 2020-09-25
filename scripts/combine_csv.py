#!venv/bin/python3

import sys, os
from argparse import ArgumentParser

def combine_csv_files(infiles, outfile, add_ids=False):
    assert infiles, 'there must be at least one file'
    header = None
    rows = []
    for infile in infiles:
        with open(infile, 'r') as f:
            lines = f.readlines()
            if header is None:
                header = lines[0]
            else:
                assert lines[0] == header, 'header rows must match in csv files'
            rows.extend(lines[1:])
    if add_ids:
        header = f'id,{header}'
        rows = [f'{i},{r}' for i,r in enumerate(rows)]
    # add new line to any rows that don't end in one.
    rows = [r if ord(r[-1]) == 10 else f'{r}\n' for r in rows]
    with open(outfile, 'w') as f:
        f.writelines(''.join([header] + rows))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-f', '--infile', nargs='+', required=True)
    parser.add_argument('-o', '--outfile', default='./combined.csv')
    args = parser.parse_args()
    combine_csv_files(args.infile, args.outfile)

# printing args for combining chunked region files:
# '-f ' + ' '.join([f'../artifacts/test/vr_{i*100}_{((i+1)*100)-1}/vregions.csv' for i in range(20)])

# ./scripts/combine_csv.py -f ./artifacts/test/vr_0_99/vregions.csv ./artifacts/test/vr_100_199/vregions.csv ./artifacts/test/vr_200_299/vregions.csv ./artifacts/test/vr_300_399/vregions.csv ./artifacts/test/vr_400_499/vregions.csv ./artifacts/test/vr_500_599/vregions.csv ./artifacts/test/vr_600_699/vregions.csv ./artifacts/test/vr_700_799/vregions.csv ./artifacts/test/vr_800_899/vregions.csv ./artifacts/test/vr_900_999/vregions.csv ./artifacts/test/vr_1000_1099/vregions.csv ./artifacts/test/vr_1100_1199/vregions.csv ./artifacts/test/vr_1200_1299/vregions.csv ./artifacts/test/vr_1300_1399/vregions.csv ./artifacts/test/vr_1400_1499/vregions.csv ./artifacts/test/vr_1500_1599/vregions.csv ./artifacts/test/vr_1600_1699/vregions.csv ./artifacts/test/vr_1700_1799/vregions.csv ./artifacts/test/vr_1800_1899/vregions.csv ./artifacts/test/vr_1900_1999/vregions.csv -o ./artifacts/test/verified_regions.csv