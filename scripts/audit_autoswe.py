"""Audit auto-swe.db for sensitive data before public release."""

import re
import sqlite3

conn = sqlite3.connect("auto-swe.db")
cur = conn.cursor()

print("=== Sensitive Pattern Scan ===\n")

patterns = {
    "API keys (sk-)": r"sk-[a-zA-Z0-9]{20,}",
    "API keys (key-)": r"key-[a-zA-Z0-9]{20,}",
    "AWS keys": r"AKIA[A-Z0-9]{16}",
    "Bearer tokens": r"Bearer [a-zA-Z0-9\-._~+/]{20,}",
    "Private keys": r"-----BEGIN.*PRIVATE KEY-----",
    "Connection strings": r"(mongodb|postgres|mysql|redis)://\S+",
    "Env var assignments": r"(OPENAI_API_KEY|ANTHROPIC_API_KEY|AWS_SECRET|DATABASE_URL|SECRET_KEY)\s*=\s*\S+",
}

cur.execute("SELECT id, input_text, output_text FROM llm_requests")
rows = cur.fetchall()
print(f"Scanning {len(rows)} rows...\n")

for name, pattern in patterns.items():
    hits = 0
    sample = None
    for row in rows:
        for text in [row[1] or "", row[2] or ""]:
            match = re.search(pattern, text)
            if match:
                hits += 1
                if sample is None:
                    sample = f"  row {row[0][:20]}: ...{match.group()[:40]}..."
    status = f"FOUND ({hits})" if hits else "clean"
    print(f"  {name}: {status}")
    if sample:
        print(f"    {sample}")

print("\n=== Project / Repo References ===\n")

cur.execute("SELECT * FROM projects")
cols = [d[0] for d in cur.description]
for r in cur.fetchall():
    d = dict(zip(cols, r))
    print(f"  Project: {d}")

print("\n=== Git References (sample) ===\n")
cur.execute("SELECT id, git_branch, git_pr_url, github_issue_url FROM issues WHERE git_pr_url IS NOT NULL LIMIT 5")
for r in cur.fetchall():
    print(f"  issue={r[0][:20]}, branch={r[1]}, pr={r[2]}, gh_issue={r[3]}")

print("\n=== Email Addresses ===\n")
email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
email_hits = 0
email_samples = set()
for row in rows:
    for text in [row[1] or "", row[2] or ""]:
        matches = re.findall(email_pattern, text)
        for m in matches:
            if m not in email_samples and len(email_samples) < 10:
                email_samples.add(m)
            email_hits += 1
print(f"  Found {email_hits} email occurrences")
if email_samples:
    print(f"  Samples: {email_samples}")

print("\n=== URLs in prompts/responses (sample) ===\n")
url_pattern = r"https?://[^\s<>\"']+"
url_samples = set()
for row in rows[:500]:
    for text in [row[1] or "", row[2] or ""]:
        for m in re.findall(url_pattern, text):
            if "github.com" not in m and "localhost" not in m and len(url_samples) < 15:
                url_samples.add(m)
if url_samples:
    for u in sorted(url_samples):
        print(f"  {u}")
else:
    print("  No non-github/localhost URLs found in first 500 rows")

print("\n=== Tables with personal/org data ===\n")
for table in ["machines", "models", "machine_models"]:
    cur.execute(f"PRAGMA table_info({table})")
    cols = [c[1] for c in cur.fetchall()]
    cur.execute(f"SELECT * FROM {table} LIMIT 3")
    print(f"{table} (cols: {cols}):")
    for r in cur.fetchall():
        print(f"  {r}")
    print()

conn.close()
