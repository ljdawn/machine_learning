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
    from sklearn import preprocessing
    target = [map(float, x) for x in np.array(data_matrix)]
    min_max_scaler = preprocessing.MinMaxScaler()
    StandardScaler = preprocessing.StandardScaler()
    model = min_max_scaler.fit(target) if mode is 0 else StandardScaler.fit(target)
    return model.transform(target)

def my_binarizer(data_matrix, threshold):
    import numpy as np
    from sklearn import preprocessing
    binarizer = preprocessing.Binarizer(threshold)
    return binarizer.transform(np.array(data_matrix))

def scale_binar_list(data_matrix, binar_thr_list = None):
    import numpy as np
    if binar_thr_list is None: binar_thr_list = [line.mean() for line in np.array(data_matrix).T]
    return my_binarizer(data_matrix, binar_thr_list)

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

if __name__ == '__main__':
    import numpy as np
    data = [[1, 2, 3, 13], [4, 5, 6, 7], [7, 8, 9, 13], [10, 11, 12, 14]]
    data_title = ['a', 'b', 'c', 'd']
    #a = pd.DataFrame({'a':[1, 2, 3], 'b':[1, 2, 3], 'c':[1, 2, 3], 'd':[1, 2, 3]})
    test = split_data_matrix(data, data_title, value_list = ['a', 'c'], binar_list = ['b'], discret_list = ['c', 'd'])
    a = scale_value_list(test.next(), 1)
    b = scale_binar_list(test.next(), [1])
    c = scale_discret_list(test.next())
    print a
    print b
    print c