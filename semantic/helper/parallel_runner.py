import os
import sys
import math
import time
import subprocess
import setproctitle

os.environ["PYTHONUNBUFFERED"] = "1"

intv = 1.0

if __name__ == '__main__':
    tot_sample, num_worker = int(sys.argv[-3]), int(sys.argv[-1])
    assert sys.argv[-4] == '--totsample' and sys.argv[-2] == '--worker'
    cmd = sys.argv[1:-4]
    cmds = [cmd + ['--start', str(i * int(math.ceil(tot_sample / num_worker))),
                   '--max', str(min((i + 1) * int(math.ceil(tot_sample / num_worker)), tot_sample))] for i in range(num_worker)]
    fins = [False for _ in range(num_worker)]

    setproctitle.setproctitle("parallel_hoster")

    procs = [subprocess.Popen(cmds[i], bufsize=0, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True) for i in range(num_worker)]

    while not all(fins):
        for i in range(num_worker):
            if not fins[i]:
                line = procs[i].stdout.readline()
                if not line:
                    fins[i] = True
                    print(f'fin proc[{i}]')
                else:
                    print(f'proc[{i}]', line.strip())
        time.sleep(intv)

    for i in range(num_worker):
        print(f'merging proc[{i}]')
        procs[i].wait()

    print('done')
