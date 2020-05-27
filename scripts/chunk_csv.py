#!venv/bin/python3

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
    parser.add_argument('-f', '--csvfile', required=True)
    parser.add_argument('-s', '--chunksize', type=int, required=True)
    parser.add_argument('-nh', '--noheader', action='store_true')
    parser.add_argument('-o', '--outdir', default='./')
    args = parser.parse_args()
    chunk_csv(args.csvfile, args.chunksize, outdir=args.outdir, with_header=not args.noheader)

