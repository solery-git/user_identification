# -*- coding: utf-8 -*-
import joblib
from pathlib import Path

def load(base_path, rel_path=None):
    path = base_path.joinpath(rel_path) if rel_path is not None else base_path
    
    with open(path, 'rb') as fin:
        obj = joblib.load(fin)
    
    return obj


def load_multiple(base_path, rel_paths):
    return [load(base_path, rel_path) for rel_path in rel_paths]


def dump(obj, base_path, rel_path=None):
    path = base_path.joinpath(rel_path) if rel_path is not None else base_path
    
    with open(path, 'wb') as fout:
        joblib.dump(obj, fout)


def dump_multiple(objs, base_path, rel_paths):
    for obj, rel_path in zip(objs, rel_paths):
        dump(obj, base_path, rel_path)