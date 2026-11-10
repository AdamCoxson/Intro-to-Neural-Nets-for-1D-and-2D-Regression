# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 11:49:35 2025

@author: acoxson
"""

from rdkit import Chem
from rdkit.Chem import (
                        AllChem,
                        rdCoordGen,
                        Draw,
                        rdFingerprintGenerator,
                        PandasTools,
                        Descriptors
                        )
import numpy as np

def getMolDescriptors(mol, descriptor_list=None, missingVal=None):
    ''' calculate the full list of descriptors for a molecule
    
        missingVal is used if the descriptor cannot be calculated
    '''
    res = {}
    if not(descriptor_list):
        descriptors = Descriptors._descList
    # TODO: Add else clause to handle a list numbers corresponding to the descriptor indices
    else:
        descriptors = [Descriptors._descList[idx] for idx in descriptor_list]

    for nm,fn in descriptors:
        # some of the descriptor fucntions can throw errors if they fail, catch those here:
        try:
            val = fn(mol)
        except:
            # print the error message:
            import traceback
            traceback.print_exc()
            # and set the descriptor value to whatever missingVal is
            val = missingVal
        res[nm] = val
    return res

# Function to categorize columns
def categorize_columns(df):
    decimal_cols = []
    integer_cols = []
    binary_cols = []

    for col in df.columns:
        unique_vals = df[col].dropna().unique()
        if np.all(df[col] == 0):  # Drop all-zero columns
            continue
        elif df[col].dtype == float:  # Decimal values
            decimal_cols.append(col)
        elif set(unique_vals).issubset({0, 1}):  # Binary columns
            binary_cols.append(col)
        else:  # Other integer columns
            integer_cols.append(col)

    return decimal_cols, integer_cols, binary_cols