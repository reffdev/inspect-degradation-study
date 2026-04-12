from datasets import load_dataset
ds = load_dataset("nebius/SWE-agent-trajectories", split="train", streaming=True)
seen = set()
for r in ds:
    iid = r["instance_id"]
    if iid in seen:
        continue
    seen.add(iid)
    n = sum(1 for m in r["trajectory"] if m.get("role") == "ai")
    print(f"{iid[:50]}: {n} steps")
    if len(seen) >= 10:
        break
