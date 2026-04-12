from datasets import load_dataset
from collections import Counter

ds = load_dataset("nebius/SWE-agent-trajectories", split="train", streaming=True)

models = Counter()
step_counts = []
exit_statuses = Counter()
traj_sample = None

for i, row in enumerate(ds):
    models[row["model_name"]] += 1
    step_counts.append(len(row["trajectory"]))
    exit_statuses[row["exit_status"]] += 1
    if traj_sample is None:
        traj_sample = row["trajectory"][:2]  # first 2 steps of first trace
    if i >= 2000:  # sample 2k rows, enough to see the shape
        break

print(f"\n=== Models ({len(models)} distinct) ===")
for model, count in models.most_common():
    print(f"  {model}: {count}")

print(f"\n=== Steps per trace ===")
import numpy as np
arr = np.array(step_counts)
print(f"  mean={arr.mean():.1f}, median={np.median(arr):.0f}, min={arr.min()}, max={arr.max()}")

print(f"\n=== Exit statuses ===")
for status, count in exit_statuses.most_common():
    print(f"  {status}: {count}")

print(f"\n=== Sample trajectory step (first 500 chars) ===")
if traj_sample:
    for j, step in enumerate(traj_sample):
        print(f"\n--- step {j} type={type(step).__name__} ---")
        print(str(step)[:500])
