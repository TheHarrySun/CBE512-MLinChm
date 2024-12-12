import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess(filename):
    # read data
    data = pd.read_csv(filename)
    
    num_X_columns = []
    nonnum_X_columns = []
    for i in range(0, len(data.columns) - 1):
        datapoint = data[data.columns[i]][1]
        if (isinstance(datapoint, int) or isinstance(datapoint, float) or isinstance(datapoint, np.int64)):
            num_X_columns.append(data.columns[i])
        else:
            nonnum_X_columns.append(data.columns[i])

    X = np.array(data[num_X_columns])
    nan_indices = np.where(np.isnan(X))
    inf_indices = np.where(np.isinf(X))
    removed = np.concatenate((nan_indices[0], inf_indices[0]), axis = 0)
    X = np.delete(X, removed, axis = 0)
    
    nonnum_X = np.array(data[nonnum_X_columns])
    Y = np.array(data[data.columns[-1]])
    Y = np.reshape(Y, (-1, 1))
    Y = np.delete(Y, removed, axis = 0)
    # standardize the data
    inscaler = StandardScaler()
    inscaler.fit(X)
    Xsc = inscaler.transform(X)

    outscaler = StandardScaler()
    outscaler.fit(Y)
    Ysc = outscaler.transform(Y)
        
    return X, Y, Xsc, Ysc, nonnum_X, removed

def main():
    ligand_data = preprocess('data_ligand.csv')
    pocket_data = preprocess('data_protein.csv')

    X = ligand_data[0]
    X_pocket = pocket_data[0]
    X_pocket = np.delete(X_pocket, ligand_data[5], axis = 0)
    Xsc = ligand_data[2]
    Xsc_pocket = pocket_data[2]
    Xsc_pocket = np.delete(Xsc_pocket, ligand_data[5], axis = 0)

    Y = ligand_data[1]
    Y_pocket = pocket_data[1]
    Y_pocket = np.delete(Y_pocket, ligand_data[5], axis = 0)
    Ysc = ligand_data[3]
    Ysc_pocket = pocket_data[3]
    Ysc_pocket = np.delete(Ysc_pocket, ligand_data[5], axis = 0)
    
    return X, Xsc, X_pocket, Xsc_pocket, Y, Ysc, Y_pocket, Ysc_pocket