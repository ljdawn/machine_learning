"""
====================
model 
working with numpy, pandas, scikit-learn, statsmodels, scipy
*standardization : z-score 
*feature_selection : Chi-squared
*mehtod : SVM, logstic, GaussianNB, DecisionTree, RandomForest, AdaBoost
**main_pandas is 3times faster
====================

"""
__author__ = """\n""".join(['Xuan Zhang'])

__version__ ="""2013-01-22"""

__all__ = ['']

#coding:utf8
from tool_box import Step_Zero_Processing_v04 as SZ
from tool_box import Step_Three_ClassificationReport as ST
from tool_box import Step_Five_RoC as SF
from tool_box import Step_Six_Cross_Validation as SS
from tool_box import Step_Seven_Grid_Search as S7
from tool_box import Step_One_Feature_Selection as SO
from sklearn import linear_model, svm, tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import numpy as np
import pandas as pd
import statsmodels.api as sm
import itertools
import json
import math
from datetime import datetime
from scipy import stats

#---tool_box common functions---
column_picker = SZ.column_picker
column_rearrange_num = SZ.column_rearrange_num
column_get_label_num = SZ.column_get_label_num
preprocess = SZ.preprocess
my_binarizer = SZ.my_binarizer
my_report = ST.my_report
my_PRC = SF.my_PRC
my_CV = SS.my_CV
my_FS = SO.my_FS	

#---main functions---
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
#with pandas	
def get_table(fn, column_name_list = [], sep = '\t'):
	target = pd.read_table(fn, names = column_name_list, sep = sep)
	return target
def main(column_list_fn_ori, column_list_fn_new, column_to_use_fn, data_file, discret_list, binar_list, binar_thr_list, stand_flag = 0, p = 0.05):
	#---process start---
	#<<step 1 -- rearrage data_matrix-- >>
	#get ori_label, new_label
	column_label_ori = get_list(column_list_fn_ori)
	column_label_new = get_list(column_list_fn_new)
	column_num_new = column_get_label_num(column_label_ori, column_label_new)
	#get ori_matrix new matrix
	data_matrix_ori = get_data_matrix(data_file)
	data_matrix_new = column_rearrange_num(data_matrix_ori, column_num_new)
	#get effective matrix
	column_to_use_list = get_list(column_to_use_fn)
	column_new_list = get_new_list(column_label_new, column_to_use_list)
	data_matrix_eff = column_picker(data_matrix_new, column_new_list)
	data_matrix_eff_float = [map(lambda x:float(x) if x != 'NULL' else 0, line) for line in data_matrix_eff.tolist()]
	#<<step 2 -- preprossing data_matrix-- >>
	#def column type
	(X, process_summary) = preprocess(data_matrix_eff_float, stand_flag = stand_flag, discret_list = discret_list, binar_list = binar_list)
	nochange_ = process_summary['no change column']
	discret_ = process_summary['discret column']
	binarized_ = process_summary['binarized column']
	cate_class = process_summary['categorize_class']
	#<<step 3 -- get flag-- >>
	y = np.array(map(float, get_One_col(data_file)))
	#<<step 4 -- feature_selection -- >>
	X_F = X - X.min() 
	X_X_filter = my_FS(X_F, y)[1]
	feature_index = range(len(X_X_filter))
	feature_selected_index = list(itertools.compress(feature_index, map(lambda x:x < p, X_X_filter)))
	X_selected = column_picker(X, feature_selected_index)
	label_before = [column_label_new[x] for x in column_new_list]
	Label_A = [label_before[x] for x in range(nochange_[0], nochange_[1])]
	Label_B = [label_before[x] for x in range(binarized_[0], binarized_[1])]
	Label_C_t = [label_before[x] for x in range(discret_[0], len(label_before))]
	Label_C = [itertools.repeat(Label_C_t[i], int(cate_class[i])) for i in range(len(cate_class))]
	Label_ALL = Label_A+Label_B
	for ite in Label_C:
		n = 0
		for it in list(ite):
			Label_ALL.append(it + '_'+str(n))
			n += 1
	timer('selected feature base on X2(p<'+str(p)+'):'+'\n'+'-'*100)
	Label_selected_ALL = [Label_ALL[x] for x in feature_selected_index]
	#timer(Label_selected_ALL) 
	#timer(len(Label_selected_ALL))
	return (X_selected, y, Label_selected_ALL)
def main_pandas(column_list_fn_ori, column_list_fn_new, column_to_use_fn, data_file, discret_list, binar_list, binar_thr_list, stand_flag = 0, p = 0.05):
	#---process start---
	column_label_ori = get_list(column_list_fn_ori)
	column_label_new = get_list(column_list_fn_new)
	column_num_new = column_get_label_num(column_label_ori, column_label_new)
	data_matrix = get_table(data_file, column_label_ori)
	column_to_use_list = get_list(column_to_use_fn)
	column_new_list = get_new_list(column_label_new, column_to_use_list)
	data_matrix_m = data_matrix[column_label_new]
	data_matrix_eff = data_matrix_m[column_new_list].values
	data_matrix_eff_float = [map(lambda x:float(x) if not math.isnan(x) else 0, line) for line in data_matrix_eff.tolist()]
	#<<step 2 -- preprossing data_matrix-- >>
	#def column type
	(X, process_summary) = preprocess(data_matrix_eff_float, stand_flag = stand_flag, discret_list = discret_list, binar_list = binar_list)
	nochange_ = process_summary['no change column']
	discret_ = process_summary['discret column']
	binarized_ = process_summary['binarized column']
	cate_class = process_summary['categorize_class']
	#<<step 3 -- get flag-- >>
	y = np.array(map(float, get_One_col(data_file)))
	#<<step 4 -- feature_selection -- >>
	X_F = X - X.min() 
	X_X_filter = my_FS(X_F, y)[1]
	feature_index = range(len(X_X_filter))
	feature_selected_index = list(itertools.compress(feature_index, map(lambda x:x < p, X_X_filter)))
	X_selected = column_picker(X, feature_selected_index)
	label_before = [column_label_new[x] for x in column_new_list]
	Label_A = [label_before[x] for x in range(nochange_[0], nochange_[1])]
	Label_B = [label_before[x] for x in range(binarized_[0], binarized_[1])]
	Label_C_t = [label_before[x] for x in range(discret_[0], len(label_before))]
	Label_C = [itertools.repeat(Label_C_t[i], int(cate_class[i])) for i in range(len(cate_class))]
	Label_ALL = Label_A+Label_B
	for ite in Label_C:
		n = 0
		for it in list(ite):
			Label_ALL.append(it + '_' + str(n))
			n += 1
	#timer('selected feature base on X2(p<'+str(p)+'):'+'\n'+'-'*100)
	Label_selected_ALL = [Label_ALL[x] for x in feature_selected_index]
	#timer(Label_selected_ALL) 
	#timer(len(Label_selected_ALL))
	return (X_selected, y, feature_selected_index)

def main_pandas_for_test(data_file, to_test, feature_selected_index, column_list_fn_ori, column_list_fn_new, column_to_use_fn, stand_flag, discret_list, binar_list, binar_thr_list):
	#---process start---
	column_label_ori = get_list(column_list_fn_ori)
	column_label_new = get_list(column_list_fn_new)
	column_num_new = column_get_label_num(column_label_ori, column_label_new)
	data_matrix_tr = get_table(data_file, column_label_ori)
	data_matrix_te = get_table(to_test, column_label_ori)
	data_matrix = pd.concat([data_matrix_tr, data_matrix_te])
	column_to_use_list = get_list(column_to_use_fn)
	column_new_list = get_new_list(column_label_new, column_to_use_list)
	data_matrix_m = data_matrix[column_label_new]
	data_matrix_eff = data_matrix_m[column_new_list].values
	data_matrix_eff_float = [map(lambda x:float(x) if not math.isnan(x) else 0, line) for line in data_matrix_eff.tolist()]
	#<<step 2 -- preprossing data_matrix-- >>
	#def column type
	(X, process_summary) = preprocess(data_matrix_eff_float, stand_flag = stand_flag, discret_list = discret_list, binar_list = binar_list)
	#<<step 3 -- get flag-- >>
	y = np.array(map(float, get_One_col(to_test)))
	#<<step 4 -- feature_selection -- >>
	X_selected = column_picker(X, feature_selected_index)[len(data_matrix_tr):]
	return (X_selected, y)

if __name__ == '__main__':
	"""loop the set_up file"""
	jsonpath = 'json/'
	fnlist = ['setup.json',]
	for fn in fnlist:
		setup = json.load(file(jsonpath + fn))
		model_flag = setup['model_flag']
		stand_flag = setup['stand_flag']
		model_M = setup['method']
		#---config/data files---
		column_list_fn_ori = setup['column_list_fn_ori']
		column_list_fn_new = setup['column_list_fn_new']
		#---model setup---
		tol =  setup['tol']
		penalty = setup['penalty']
		C = setup['C']
		p = setup['P']
		logstic_threshold = setup['logstic_threshold']
		cv_fold	= setup['cv_fold']
		#function switch
		ROC_plot = setup['ROC_plot']
		CV_score = setup['CV_score']
		#---customize------------------------------------->>>>>>
		column_to_use_fn = setup['model_detailed'][model_flag]['column_to_use_fn']
		data_file = setup['model_detailed'][model_flag]['data_file']
		data_file_test = setup['model_detailed'][model_flag]['data_file_test']
		discret_list = setup['model_detailed'][model_flag]['discret_list']
		binar_list = setup['model_detailed'][model_flag]['binar_list']
		binar_thr_list = setup['model_detailed'][model_flag]['binar_thr_list']
		#---customize------------------------------------->>>>>>
		start_time = datetime.now()
		#term 0 ==================================================
		#mining
		#column_label_ori = get_list(column_list_fn_ori)
		#pre_data = get_table(data_file, column_label_ori)
		#tclo = np.array(pre_data['inpg_ays'] - pre_data['inpg_ays'].mean())
		#tclo = tclo[np.where(tclo != -1)]
		#t-test mean test (t-value, p-value, two_sided p < 0.05 -> mean -> 0)
		#res_sm = sm.stats.DescrStatsW(tclo).ztest_mean()
		#res_st = stats.ttest_1samp(tclo, 0.0)
		#print res_sm, res_st
		#vat - test
		#term 0 ==================================================
	
		#term 1-1 ==================================================
		"""data processing : 1, origin data column (order) rearragement(optional) -> to fit the format. <continuous>:<binary>:<discrest box> 
		                2, column picking(optional) -> to fit the model.
						3, processing data -> to fit the format for machine learning(training part).
						4, feature selection -> to fit the model.
						5, get the <y>s. -> to fit the format for machine learning(training part).
						"""
		#(X_selected, y) = main(column_list_fn_ori = column_list_fn_ori, column_list_fn_new = column_list_fn_new, column_to_use_fn = column_to_use_fn, data_file = data_file, \
		#	stand_flag = stand_flag, discret_list = discret_list, binar_list = binar_list, binar_thr_list = binar_thr_list, p=p)
		#print (X_selected, y)
		#term 1-1 ==================================================
	
		#term 1-2 ==================================================
		"""data processing : 1, origin data column (order) rearragement(optional) -> to fit the format. <continuous>:<binary>:<discrest box> 
			                2, column picking(optional) -> to fit the model.
							3, processing data -> to fit the format for machine learning(training part).
							4, feature selection -> to fit the model.
							5, get the <y>s. -> to fit the format for machine learning(training part).
							"""
		(X_selected, y, F_selected) = main_pandas(column_list_fn_ori = column_list_fn_ori, column_list_fn_new = column_list_fn_new, column_to_use_fn = column_to_use_fn, data_file = data_file, \
			stand_flag = stand_flag, discret_list = discret_list, binar_list = binar_list, binar_thr_list = binar_thr_list, p = p)
		#term 1-2 ==================================================
	
		#term 2 ==================================================
		timer('<< -- training '+ model_M +' -- >>')
		"""training model: 1, y_ -> predicted values 2, -> predicted values in probility"""
		if model_M == 'logstic':
			M = linear_model.LogisticRegression(tol = tol, penalty = penalty, C = C)
		elif model_M == 'SVM':
			M = svm.SVC(C = C, tol = tol, kernel='linear', probability = True)
		elif model_M == 'GaussianNB':
			M = GaussianNB()
		elif model_M == 'DecisionTree':
			M = tree.DecisionTreeClassifier()
		elif model_M == 'RandomForest':
			M = RandomForestClassifier(n_estimators = len(y))
		elif model_M == 'AdaBoost':
			M = AdaBoostClassifier(n_estimators = len(y))
		Model = M.fit(X_selected, y)
		"""predicting"""
		for da_t in data_file_test:
			timer('<< -- predicting '+ da_t +' -- >>')
			(X_selected_test, y_test) = main_pandas_for_test(data_file = data_file, to_test = da_t, feature_selected_index = F_selected, column_list_fn_ori = column_list_fn_ori, column_list_fn_new = column_list_fn_new, column_to_use_fn = column_to_use_fn,\
				stand_flag = stand_flag, discret_list = discret_list, binar_list = binar_list, binar_thr_list = binar_thr_list)
			y_ori = Model.predict(X_selected_test)
			y_p = [b for [a, b] in Model.predict_proba(X_selected_test)]
			y_ = my_binarizer(y_p, logstic_threshold)
			y = y_test
			#print y, y_
			#term 2 ==================================================
			
			#term 3 ==================================================
			print 'confusion_matrix:'
			"""confusion_matrix"""
			print 'CMAT', '[0]','----', '[1]'
			print '[0]', '|', my_report(y,y_)[0][0][0], '\t', my_report(y,y_)[0][0][1], '|' 
			print '[1]', '|', my_report(y,y_)[0][1][0], '\t', my_report(y,y_)[0][1][1], '|'
			"""plot ROC curve only in windows"""
			if ROC_plot == 'enable':
				import pylab as pl
				pl.clf()
				pl.plot(my_PRC(y,y_)[0], my_PRC(y,y_)[1], label='ROC curve (area = %0.2f)' % my_PRC(y,y_)[3])
				pl.plot([0, 1], [0, 1], 'k--')
				pl.xlim([0.0, 1.0])
				pl.ylim([0.0, 1.0])
				pl.xlabel('False Positive Rate')
				pl.ylabel('True Positive Rate')
				pl.title('Receiver operating characteristic example')
				pl.legend(loc="lower right")
				pl.show()
			#term 3 ==================================================
		
			#term 4 ==================================================
			print 'summary report:'
			"""detailed report"""
			print my_report(y,y_)[1]
			#print '\nROC curve area:','\n','-'*100
			#"""ROC curve"""
			#print my_PRC(map(int, y.tolist()), y_p)[3]
			#term 4 ==================================================
		
			#term 5 ==================================================
			if CV_score == 'enable':
				print '\nCV_score :','\n'
				func = M
				print my_CV(X_selected, y, func, cv_fold)
			#term 5 ==================================================
			end_time =  datetime.now()
			print 'time_cost:', str(end_time - start_time)
			print 'current Model:', model_flag, '\n', 'current Classifier:', model_M, '\n','-'*100, '\n'