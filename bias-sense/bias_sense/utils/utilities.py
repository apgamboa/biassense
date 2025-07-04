import os
import pickle


def load_pickle(path, file_name):
    full_path = path / file_name
    if os.path.isfile(full_path):
        with open(full_path, "rb") as f:
            try:
                return pickle.load(f)
            except Exception: # so many things could go wrong, can't be more specific.
                pass
    return None

def save_pickle(path, file_name, transformer_model):
    full_path = path / file_name
    with open(full_path, "wb") as f:
        pickle.dump(transformer_model, f)
    return transformer_model