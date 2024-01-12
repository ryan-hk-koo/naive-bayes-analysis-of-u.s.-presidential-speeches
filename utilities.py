# -*- coding: utf-8 -*-

import pickle
import pandas as pd

# Saving URL list, etc.
def dump(path, name):
    file = open(path,"wb")
    pickle.dump(name, file)
    file.close()
    
# Loading URL list, etc.
def load(path):
    file = open(path, "rb")
    data = pickle.load(file)
    file.close()
    return data

    

