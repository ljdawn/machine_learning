"""
====================
working with numpy, pandas, scikit-learn, scipy
*feature_selection : Chi-squared
*mehtod : logstic
*missing values strategy updated -> mean, median, most_frequent
**cost sensitive learning added 
**grid searching for (L1, L2, C)  added
====================

"""
__author__ = """\n""".join(['Xuan Zhang'])

__version__ ="""2014-03-03"""

__all__ = ['']
#coding:utf8
from tool_box import Step_Zero_Processing_v08 as SZ
from tool_box import Step_Three_ClassificationReport as ST
from tool_box import Step_Five_RoC as SF
from tool_box import Step_Six_Cross_Validation as SS
from tool_box import Step_Seven_Grid_Search as S7
from tool_box import Step_One_Feature_Selection as SO
from sklearn import linear_model
from sklearn import grid_search
from sklearn.preprocessing import Imputer
from datetime import datetime
from scipy import stats
import numpy as np
import pandas as pd
import itertools
import json
import math
import logging


#---tool_box common functions---
get_data_matrix = SZ.get_data_matrix
get_list = SZ.get_list
get_prepared_data_matrix = SZ.get_prepared_data_matrix
get_One_col = SZ.get_One_col
column_picker = SZ.column_picker
my_FS = SO.my_FS

def process(set_up, predict_flag = 0):
    import pandas as pd
    import numpy as np
    import cPickle as pickle
    imp = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
    setup = json.load(file(set_up))
    ori_col_names = get_list(setup['column_list_fn_ori'])
    mode = setup['model_flag']
    data_file_tr = np.array(get_data_matrix(setup['model_detailed'][mode]['data_file']))
    data_file_te = np.array(get_data_matrix(setup['model_detailed'][mode]['data_file_test']))
    data_file = np.vstack([data_file_tr, data_file_te])
    discret_col = get_list(setup['model_detailed'][mode]['discret_list'])
    col_to_use = get_list(setup['model_detailed'][mode]['column_to_use_fn'])
    value_list = [colname for colname in col_to_use if colname not in discret_col]
    data_matrix = np.array(pd.DataFrame(data_file, columns = ori_col_names)[col_to_use].values)
    data_matrix = [map(lambda ele:float(ele) if ele != 'NULL'  else -1.0, line) for line in data_matrix]
    data_matrix = imp.fit(data_matrix).transform(data_matrix)
    prepared_data = get_prepared_data_matrix(data_matrix, col_to_use, value_list = value_list, discret_list = discret_col)
    X, X_ = prepared_data.values[:len(data_file_tr)], prepared_data[len(data_file_tr):]
    y = np.array(map(float, get_One_col(setup['model_detailed'][mode]['data_file'])))
    X_F = X - X.min()
    X_X_filter = my_FS(X_F, y)[1]
    feature_index = range(len(X_X_filter))
    feature_selected_index = list(itertools.compress(feature_index, map(lambda x:x < float(setup['P']), X_X_filter)))
    X_selected = column_picker(X, feature_selected_index)
    X_totest = column_picker(X_, feature_selected_index)
    feature_selected_label = np.array(prepared_data.keys())[feature_selected_index]
    pickle.dump(feature_selected_label, open(setup['feature_saving_path'], 'w'))
    return (pd.DataFrame(X_selected, columns = feature_selected_label), pd.DataFrame(X_totest, columns = feature_selected_label))

if __name__ == '__main__':
    process('json/setup.json')