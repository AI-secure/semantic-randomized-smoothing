import os
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='Analyze the real performance from logs')
parser.add_argument("logfile", help="path of the certify.py output")
parser.add_argument("outfile", help="the output path of the report")
parser.add_argument("--budget", type=float, default=0.0,
                    help="for semantic certification, the pre-allocated space for semantic transformations")
parser.add_argument("--step", type=float, default=0.25, help="step size for l2 robustness")
args = parser.parse_args()

if __name__ == '__main__':
    df = pd.read_csv(args.logfile, delimiter="\t")
    print(f'Total: {len(df)} records')
    steps = list()
    nums = list()
    now_step = args.budget
    while True:
        cnt = (df["correct"] & (df["radius"] >= now_step)).sum()
        mean = (df["correct"] & (df["radius"] >= now_step)).mean()
        steps.append(now_step)
        nums.append(mean)
        now_step += args.step
        if cnt == 0:
            break
    steps = [str(s) for s in steps]
    nums = [str(s) for s in nums]
    output = "\t".join(steps) + "\n" + "\t".join(nums)
    print(output)
    # print(f'Output to {args.outfile}')
    # f = open(args.outfile, 'w')
    # print(output, file=f)
    print(f'Clean acc: {df["correct"].sum()}/{len(df)} = {df["correct"].sum()/len(df)}')
    # f.close()
