from curses import raw
import numpy as np
import pandas as pd
import scanpy as sc
import copy

def hvg(adata, min_mean=0.05, max_mean=5):
    """
    Select highly expressed genes (not real HVGs, only mean filter).
    """
    data = np.array(adata.X.mean(axis=0)).ravel()
    data = pd.Series(data, index=adata.var_names)

    # 筛选条件
    index = data[(data > min_mean) & (data < max_mean)].index

    # 标记
    adata.var['highly_variable'] = False
    adata.var.loc[index, 'highly_variable'] = True

    return adata
