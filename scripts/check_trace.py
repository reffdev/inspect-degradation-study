from datasets import load_dataset
ds = load_dataset("nebius/SWE-agent-trajectories", split="train", streaming=True)
rows = list(zip(range(10), ds))
r = rows[9][1]
n_ai = sum(1 for m in r["trajectory"] if m.get("role") == "ai")
print(f"instance={r['instance_id']}, model={r['model_name']}, traj_len={len(r['trajectory'])}, ai_steps={n_ai}")
