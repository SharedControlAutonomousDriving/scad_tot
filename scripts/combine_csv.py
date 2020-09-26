#!venv/bin/python3

"""
combine_csv.py

Combines CSV files into a single CSV, and handles headers

Examples:
help
./scripts/combine_csv.py -h

combine a.csv and b.csv into ab.csv
./scripts/combine_csv.py -f a.csv b.csv -o ab.csv

combine a.csv and b.csv (without header rows) into ab.csv
./scripts/combine_csv.py -f a.csv b.csv -o ab.csv -nh

combine a.csv and b.csv into ab.csv, and add 'id' column
./scripts/combine_csv.py -f a.csv b.csv -o ab.csv -id
"""

import sys, os
from argparse import ArgumentParser

def combine_csv_files(infiles, outfile, noheader=False, idcol=False):
    assert infiles, 'there must be at least one input file'
    header, rows = None, []
    for i, infile in enumerate(infiles):
        with open(infile, 'r') as f:
            lines = f.readlines()
            if i == 0:
                header = lines[0]
            elif not noheader:
                assert lines[0] == header, 'header rows must match in csv files'
            rows.extend(lines[0 if noheader else 1:])
    if idcol:
        header = f'id,{header}'
        rows = [f'{i},{r}' for i,r in enumerate(rows)]
    # add new line to any rows that don't end in one.
    rows = [r if ord(r[-1]) == 10 else f'{r}\n' for r in rows]
    with open(outfile, 'w') as f:
        f.writelines(''.join(([header] if not noheader else []) + rows))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-f', '--infile', nargs='+', required=True, help='input csv files to combine')
    parser.add_argument('-o', '--outfile', default='./combined.csv', help='specify output filename')
    parser.add_argument('-nh', '--noheader', action='store_true', help='use if input csv does not have header row')
    parser.add_argument('-id', '--idcol', action='store_true', help='add id column to combined csv')
    args = parser.parse_args()
    combine_csv_files(args.infile, args.outfile, noheader=args.noheader, idcol=args.idcol)
