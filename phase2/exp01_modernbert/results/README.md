# results/ — partials mirrored from the Colab VM

`results.jsonl` is written on the Colab VM at `/content/exp01/results.jsonl`,
one JSON line appended **after each run completes**. This directory holds
snapshots pulled down from the VM as runs land.

**Absence of a file here does not mean the sweep failed.** It means no run has
completed since the last pull. Check `run_count.txt` for what the last pull saw.

Each pull overwrites `results.jsonl` with the VM's current full contents (the
file only ever grows by appending, so a later snapshot is a superset of an
earlier one). `run_count.txt` records how many of the 60 runs the snapshot
contains and when it was taken.

A snapshot with fewer than 60 lines is PARTIAL. `code/aggregate.py` marks any
cell with fewer than 3 seeds as PARTIAL rather than averaging over whatever
happens to be present.
