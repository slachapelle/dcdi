import os
import pickle
import uuid

import pandas as pd


def dump(obj, exp_path, name, txt=False):
    if not txt:
        with open(os.path.join(exp_path, name + ".pkl"), "wb") as f:
            pickle.dump(obj, f)
    else:
        with open(os.path.join(exp_path, name + ".txt"), "w") as f:
            f.write(str(obj))


def load(exp_path, name):
    with open(os.path.join(exp_path, name), "rb") as f:
        obj = pickle.load(f)
    return obj


def np_to_csv(array, save_path):
    """
    Convert np array to .csv

    array: numpy array
        the numpy array to convert to csv
    save_path: str
        where to temporarily save the csv
    Return the path to the csv file
    """
    id = str(uuid.uuid4())
    output = os.path.join(os.path.dirname(save_path), 'tmp_' + id + '.csv')

    df = pd.DataFrame(array)
    df.to_csv(output, header=False, index=False)

    return output

