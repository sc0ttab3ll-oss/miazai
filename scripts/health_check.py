from pathlib import Path
import importlib

checks = []

def ok(name):
    checks.append(f"OK  - {name}")

def fail(name, err):
    checks.append(f"ERR - {name}: {err}")

project_root = Path("/home/scotty/miazai")

if project_root.exists():
    ok("project root exists")
else:
    fail("project root exists", "missing /home/scotty/miazai")

for name in ["flask", "dotenv", "pydantic", "requests", "pyphen"]:
    try:
        importlib.import_module(name)
        ok(f"python package: {name}")
    except Exception as e:
        fail(f"python package: {name}", e)

for rel in [
    "whiteear",
    "whiteego",
    "orchestrator",
    "AGENTS.md",
    "GEMINI.md",
    ".env",
]:
    p = project_root / rel
    if p.exists():
        ok(f"path: {rel}")
    else:
        fail(f"path: {rel}", "missing")

print("\nMIa-zAI v1-lite health check\n")
for line in checks:
    print(line)
