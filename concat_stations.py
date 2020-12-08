"""
Concatenates Stations
"""
import os
import pandas as pd
import glob

def concat_files():
    """returns concatenated stations"""
    path = os.getcwd()
    path = os.path.join(path, "morebikes2020/Train/Train")
    all_files = glob.glob(path + "/*.csv")
    li = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)
    
    frame = pd.concat(li, axis=0, ignore_index=True)
    return frame

