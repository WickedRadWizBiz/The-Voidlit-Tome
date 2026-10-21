"""Make JAX (installed in the reconcile venv) importable from the game venv (same Python 3.13, same numpy)."""
import sys
J = "/tmp/claude-0/-home-user-The-Voidlit-Tome/e59d6209-590c-56a8-b6eb-39e9e8d9d765/scratchpad/reconcile/jaxvenv/lib/python3.13/site-packages"
if J not in sys.path:
    sys.path.append(J)
