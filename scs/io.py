"""Save/load run outputs: one npz with arrays plus a JSON string of metadata."""
import json
import numpy as np


def save_run(path, arrays, meta):
    np.savez(path, __meta__=json.dumps(meta), **arrays)


def load_run(path):
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d["__meta__"]))
    arrays = {k: d[k] for k in d.files if k != "__meta__"}
    return arrays, meta
