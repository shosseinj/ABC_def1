import importlib, sys

required = ["numpy","sklearn","torch","pennylane","pytest"]
failed = []
print("Python:", sys.version)
for name in required:
    try:
        m = importlib.import_module(name)
        print(f"[OK] {name}: {getattr(m,'__version__','installed')}")
    except Exception as e:
        failed.append((name,str(e)))
        print(f"[FAIL] {name}: {e}")

if failed:
    raise SystemExit("Environment incomplete. Run: pip install -r requirements.txt")
print("\nPHASE 0 PASSED. You can proceed to PHASE 1.")
