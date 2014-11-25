#cording:utf-8

import prepro.colpre as preprocessing
import prepro.datapre as dreader
import prepro.trainer as trainer
import prepro.datagetter as datagetter
import numpy as np
import pandas as pd

if __name__ == '__main__':

	#setup
	pre_data_path = 'data/pre_data'
	data_path = 'data/pre_data'
	columns_path = 'data/pre_header'
	sep = ','

	#data
	data_g = datagetter.Datagetter()
	data_g.get_response()
	data_g.save_data(pre_data_path)


	#reading
	mydata = dreader.Mydata(data_path, columns_path, sep)
	data, columns = mydata.load_file()
	df_noun = data[columns['noun']]
	df_num = data[columns['num']]
	Y = data[columns['Y']]


	#preprocessing
	dfnoun = preprocessing.Colnoun(df_noun + 1)
	dfnoun.one_hot()
	noun_part = dfnoun.get_data()

	
	dfnum = preprocessing.Colnum(df_num)
	dfnum.imp()
	dfnum.scale()
	numpart = dfnum.get_data()

	all_part = preprocessing.Colcombine(Y, noun_part, numpart)
	predata = all_part.get_data()
	print predata

	#training
	new_model = trainer.Trainer(predata)
	new_model.train()
	new_model.save_model()
	new_model.show_res()
	new_model.show_report()




