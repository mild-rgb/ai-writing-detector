"""Sample-size calculator for the judge evaluation.

Two variance components matter:

  trial   -- Bernoulli, p(1-p) per trial, shrinks as 1/n
  judge   -- each judge agent instance has its own bias; shared by every trial
             it scores, so it does NOT shrink with n, only with replicates R

Because the judge component is common to a whole batch, adding trials to a
single judge's batch buys almost nothing past a point. Replicates are what buy
precision. This script reports the (R, n) grid.
"""
import math
import sys

Z_A, Z_B = 1.959964, 0.841621          # alpha .05 two-sided, power .80
P = 0.50                                # null: judge at chance


def se(R, n, sd_judge):
    """SE of the mean accuracy over R judges x n trials, in percentage points."""
    return math.sqrt(sd_judge ** 2 / R + (P * (1 - P) * 100 ** 2) / (n * R))


def min_detectable(R, n, sd_judge):
    return (Z_A + Z_B) * se(R, n, sd_judge)


if __name__ == "__main__":
    sd_judge = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    target = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    print(f"judge-instance SD = {sd_judge:.1f} pts | target detectable "
          f"difference from chance = {target:.1f} pts | alpha .05, power .80\n")
    print(f"{'R judges':>9} " + "".join(f"{('n=' + str(n)):>9}" for n in (20, 30, 50, 100, 200)))
    print("-" * 58)
    for R in (1, 2, 3, 5, 8, 12, 20):
        row = f"{R:>9} "
        for n in (20, 30, 50, 100, 200):
            mdd = min_detectable(R, n, sd_judge)
            mark = "*" if mdd <= target else " "
            row += f"{mdd:>8.1f}{mark}"
        print(row)
    print("\n* = meets the target. Total judged trials = R x n.")
    print("\nCheapest configurations meeting the target:")
    best = []
    for R in range(1, 41):
        for n in range(10, 401, 10):
            if min_detectable(R, n, sd_judge) <= target:
                best.append((R * n, R, n))
    best.sort()
    for tot, R, n in best[:5]:
        print(f"   R={R:<3} judges x n={n:<4} trials = {tot:<5} total  "
              f"(MDD {min_detectable(R, n, sd_judge):.1f} pts)")
