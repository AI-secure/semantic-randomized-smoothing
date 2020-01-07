import argparse
import re
import os

parser = argparse.ArgumentParser(description='Concatenate existing aliasing analysis or certify results')
parser.add_argument('path', type=str, help='path to pattern files, excluding the pattern')
parser.add_argument('--outfile', type=str, help='output file', default=None)
args = parser.parse_args()

if __name__ == '__main__':
    outfile = args.outfile if args.outfile is not None else args.path
    path = args.path
    dirname = os.path.dirname(path)
    filename = os.path.basename(path)
    strs = list()
    header = None
    for fname in os.listdir(dirname):
        if re.match(f'^{filename}_start_[0-9]+_end_[0-9]+$', fname) is not None:
            with open(os.path.join(dirname, fname), 'r') as f:
                print(f'joining {fname}')
                all = f.readlines()
                header, now_strs = all[0], all[1:]
                strs.extend(now_strs)
    with open(outfile, 'w') as f:
        # print('no.\tmaxl2sqr', file=f, flush=True)
        f.writelines([header] + strs)
