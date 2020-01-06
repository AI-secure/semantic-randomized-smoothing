import argparse
import re
import os

parser = argparse.ArgumentParser(description='Concatenate existing aliasing analysis results')
parser.add_argument('path', type=str, help='path to pattern files, excluding the pattern')
parser.add_argument('--outfile', type=str, help='output file', default=None)
args = parser.parse_args()

if __name__ == '__main__':
    outfile = args.outfile if args.outfile is not None else args.path
    path = args.path
    dirname = os.path.dirname(path)
    filename = os.path.basename(path)
    strs = list()
    for fname in os.listdir(dirname):
        if re.match(f'^{filename}_start_[0-9]+_end_[0-9]+$', fname) is not None:
            with open(os.path.join(dirname, fname), 'r') as f:
                print(f'joining {fname}')
                strs += f.readlines()[1:]
    with open(outfile, 'w') as f:
        print('no.\tmaxl2sqr', file=f, flush=True)
        f.writelines(strs)
