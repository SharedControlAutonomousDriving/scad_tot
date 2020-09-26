#!venv/bin/python3

"""
chunk_csv.py

Splits a CSV into chunks and handles the headers

Examples:
help
./scripts/chunk_csv.py -h

split input.csv (with header row) into chunks of 100 rows
./scripts/chunk_csv.py -f input.csv -s 100 -o ./outdir

split input.csv (without header row) into chunks of 10 rows
./scripts/chunk_csv.py -f input.csv -s 100 -nh -o ./outdir
"""

import sys, os
from argparse import ArgumentParser

def chunk(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]

def chunk_csv(csv_path, chunk_size, outdir='./', with_header=True):
    csv_fname = os.path.basename(csv_path)
    name = ''.join(csv_fname.rsplit('.csv', 1))
    rows = open(csv_path, 'r').readlines()
    hrows = []
    if with_header:
        hrows = [rows[0]]
        rows = rows[1:]
    for i,c in enumerate(chunk(rows, chunk_size)):
        c_fname = f'{name}_{i}.csv'
        c_path = os.path.join(outdir, c_fname)
        open(c_path, 'w+').writelines(hrows + c)
        print(f'wrote chunk {i} to {c_path}')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-f', '--csvfile', required=True, help='input csv file')
    parser.add_argument('-s', '--chunksize', type=int, required=True, help='size of output csv chunks')
    parser.add_argument('-nh', '--noheader', action='store_true', help='use if input csv does not have header row')
    parser.add_argument('-o', '--outdir', default='./', help='output directory to chunked csv files to')
    args = parser.parse_args()
    chunk_csv(args.csvfile, args.chunksize, outdir=args.outdir, with_header=not args.noheader)
