import numpy as np

maxruns = 1000

with open("hitting_times.txt", "w") as f:
    f.write("#1st_hit \t 2nd_hit\n")

    for run_i_m1 in range(maxruns):
        run_i = run_i_m1 + 1

        data = np.loadtxt("two_first_attached_time.{}.txt".format(run_i))

        one_hit = data[0]
        two_hit = data[1]

       # print("Run {}: 1 hit: {} 2 hit: {}".format(run_i, one_hit, two_hit))
        f.write("{}\t{}\n".format(one_hit, two_hit))
        
