"""
====================
data processing 
working with numpy, scipy, pandas, scikit-learn, statsmodels
====================

"""
__author__ = """\n""".join(['Xuan Zhang'])

__version__ = """2013-02-28"""

__all__ = ['']

#coding:utf8

def split_data_matrix(data_matrix, colnames = None, value_list = None, binar_list = None, discret_list = None):
    import pandas as pd
    if colnames is None: colnames = range(len(data_matrix[0]))
    df_data = pd.DataFrame(data_matrix, columns = colnames)
    if value_list is not None: yield df_data[value_list]
    if binar_list is not None: yield df_data[binar_list]
    if discret_list is not None: yield df_data[discret_list]

def scale_value_list(data_matrix, mode = 0):
    import numpy as np
    import pandas as pd
    from sklearn import preprocessing
    target = [map(float, x) for x in np.array(data_matrix)]
    min_max_scaler = preprocessing.MinMaxScaler()
    StandardScaler = preprocessing.StandardScaler()
    model = min_max_scaler.fit(target) if mode is 0 else StandardScaler.fit(target)
    return pd.DataFrame(model.transform(target), columns = data_matrix.keys())

def my_binarizer(data_matrix, threshold):
    import numpy as np
    from sklearn import preprocessing
    binarizer = preprocessing.Binarizer(threshold)
    return binarizer.transform(np.array(data_matrix))

def scale_binar_list(data_matrix, binar_thr_list = None):
    import numpy as np
    import pandas as pd
    if binar_thr_list is None: binar_thr_list = [line.mean() for line in np.array(data_matrix).T]
    return pd.DataFrame(my_binarizer(data_matrix, binar_thr_list), columns = data_matrix.keys())

def scale_discret_list(data_matrix):
    import numpy as np
    import pandas as pd
    import itertools
    from sklearn import preprocessing
    matrix_catalog_converter = preprocessing.OneHotEncoder()
    counter = itertools.count(0)
    title_ele = iter([[(x, ele) for ele in data_matrix[x].values] for x in data_matrix.keys()])
    title_ele_set = []
    [[title_ele_set.append(si_title) for si_title in title_line if si_title not in title_ele_set] for title_line in title_ele]
    target = [map(float, x) for x in np.array(data_matrix)]
    model = matrix_catalog_converter.fit(target)
    return pd.DataFrame(model.transform(target).toarray(), columns = title_ele_set)

def get_prepared_data_matrix(data_matrix, colnames = None, value_list = None, mode = 0, binar_list = None, discret_list = None, binar_thr_list = None):
    import itertools
    discriminant = [1 if func is not None else 0 for func in (value_list, binar_list, discret_list)]
    target = split_data_matrix(data_matrix, colnames, value_list, binar_list, discret_list)
    functions = itertools.compress([scale_value_list, scale_binar_list, scale_discret_list], discriminant)
    if sum(discriminant) == 3: res = functions.next()(target.next()).join(functions.next()(target.next())).join(functions.next()(target.next())) 
    if sum(discriminant) == 2: res = functions.next()(target.next()).join(functions.next()(target.next()))    
    if sum(discriminant) == 1: res = functions.next()(target.next())
    return res

def column_picker(data_matrix, column_to_pick):
    import numpy as np
    target = np.array(data_matrix)
    assert max(column_to_pick) < target.shape[1]
    assert np.array(column_to_pick).min() >= 0
    target_step_1 = filter(lambda (x,y):x in column_to_pick, enumerate(target.T))
    return np.vstack([x[1].T for x in target_step_1]).T

def column_rearrange_num(data_matrix, new_order):
    import numpy as np
    target = np.array(data_matrix)
    (m, n) = target.shape
    assert n == len(set(new_order))
    assert n == np.array(new_order).max() + 1
    assert np.array(new_order).min() >= 0
    new_order_dict = dict([(y, x) for (x, y) in enumerate(new_order)]) 
    ori_list_dict = dict(enumerate(np.array(data_matrix).T))
    new_matrix_list = [(new_order_dict[i], ori_list_dict[i]) for i in xrange(len(new_order))]
    new_matrix_list.sort()
    return np.array([y for (x, y) in new_matrix_list]).T

def column_get_label_num(ori_label, new_label):
    ori_label_dict = dict([(y, x) for (x, y) in enumerate(ori_label)])
    return [ori_label_dict[x] for x in new_label]

def column_interchange(dataframe_ori, dataframe_new):
    """df of ori file, df of column_to_interchange"""   
    import pandas as pd
    import numpy as np
    inter_change_key = [key for key in dataframe_ori if key in dataframe_new]
    ori_remain_key = [key for key in dataframe_ori if key not in inter_change_key]
    new_matrix = np.vstack((np.array(dataframe_ori[ori_remain_key]).T, np.array(dataframe_new[inter_change_key]).T)).T 
    new_df = pd.DataFrame(new_matrix, columns = ori_remain_key + inter_change_key)
    return new_df[dataframe_ori.keys()].values

def get_list(fn):
    with open(fn) as column_list_f:
        return map(lambda x:x.strip(), column_list_f.readlines())

def get_new_list(base, filter_):
    return map(lambda (x, y):x, filter(lambda (x, y):y in filter_, [(x, y) for (x,y) in enumerate(base)]))

def get_data_matrix(fn, sep = '\t'):
    res = []
    with open(fn) as mat:
        return map(lambda x:x.strip().split(sep), mat.readlines())

def get_new_label(colum_label_dict, filter_):
    return [colum_label_dict[x] for x in filter_]

def get_One_col(fn, col = -1):
    with open(fn) as column_One:
        return map(lambda x:x.strip()[col], column_One.readlines())
def timer(s = ''):
    print str(datetime.now()), '>>', s

def get_table(fn, column_name_list = [], sep = '\t'):
    target = pd.read_table(fn, names = column_name_list, sep = sep)
    return target

if __name__ == '__main__':
    import warnings
    import numpy as np
    import pandas as pd
    import cProfile, pstats
    import logging
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="pandas", lineno=570)
    
    data = [[1, 2, 3, 13], [4, 5, 6, 7], [7, 8, 9, 13], [10, 11, 12, 14]]
    data_title = ['a', 'b', 'c', 'd']

    #cProfile.run("get_prepared_data_matrix(data_matrix = data, colnames = data_title, value_list = ['a', 'c'], discret_list = ['c', 'd'])", "../profile/res")
    #p = pstats.Stats("../profile/res")
    #p.sort_stats("time").print_stats()
    data_preed = get_prepared_data_matrix(data_matrix = data, colnames = data_title, value_list = ['a', 'c'], discret_list = ['c', 'd'])
    print data_preed




