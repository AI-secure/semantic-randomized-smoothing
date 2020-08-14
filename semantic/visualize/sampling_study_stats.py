import pandas as pd
from dateutil import parser

NRs = [[100, 100], [100, 1000], [1000, 100], [1000, 1000], [1000, 10000], [10000, 1000]]
sigmas = ['0.05', '0.10', '0.15']

def conv_times(frame):
    return [parser.parse(frame.loc[i]['time']).time().hour * 3600.0 +
            parser.parse(frame.loc[i]['time']).time().minute * 60.0 +
            parser.parse(frame.loc[i]['time']).time().second +
            parser.parse(frame.loc[i]['time']).time().microsecond / 1E6
            for i in range(len(frame))]

if __name__ == '__main__':
    table = list()
    for N, R in NRs:
        df = pd.read_csv(f'data/data/rotation_alias_study/cifar10/{N}_{R}_p_10', delimiter='\t')
        n = len(df)
        row = [f'$N={N},R={R}$']
        row.append(f'${df["maxl2sqr"].mean():.3}$')
        row.append('$\SI{' + f'{sum(conv_times(df))/n:.4}' + '}{s}$')
        for sigma in sigmas:
            df = pd.read_csv(f'data/data/rotation_alias_study_cifar10/noise_{sigma}_{N}_{R}_p_10', delimiter='\t')
            robacc = (df["correct"] & (df["radius"] >= 0.0)).sum()/n
            row.append(f'${int(robacc*100)}\%$')
            row.append('$\SI{' + f'{sum(conv_times(df))/n:.3}' + '}{s}$')
        table.append(row)

    toprule = '\\toprule\n'
    head = '\multirow{2}{*}{Sampling Numbers} & Sampling & Computing & ' + ' & '.join(['\multicolumn{2}{c}{$\\sigma=' + s + '$}' for s in sigmas]) + '\\\\\n'
    head2 = '& Err. $M$ & Time & ' + ' & '.join(['Rob. Acc. & Certify Time' for _ in sigmas]) + '\\\\\n'
    hline = '\\hline\n'
    body = ''.join([(' & '.join(row)) + '\\\\\n' for row in table])
    bottomline = '\\bottomrule\n'

    table_str = toprule + head + head2 + hline + body + bottomline
    print(table_str)

