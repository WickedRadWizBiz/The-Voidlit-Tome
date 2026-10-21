import os, sys, time
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
t0 = time.perf_counter()
import jaxpath  # noqa
import jax
if len(sys.argv) > 1:
    jax.config.update("jax_compilation_cache_dir", sys.argv[1])
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
import numpy as np, torch
import cosim as CS
t1 = time.perf_counter()
T = CS.Tome(seed=0)
t2 = time.perf_counter()
T.set_game_inputs(np.zeros(64, np.float32), 1.0)
T.run_turn(ms=8)
t3 = time.perf_counter()
print(f"cache={'on' if len(sys.argv) > 1 else 'off'}: imports {t1-t0:.2f}s, build {t2-t1:.2f}s, first window incl. JIT {t3-t2:.2f}s")
