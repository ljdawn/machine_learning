#cording:utf-8

import datetime
import urllib2
import cPickle as pickle
import json
import re
import operator
import numpy as np
from scipy.stats import poisson

class Datagetter(object):
	"""docstring for Datagetter"""
	def __init__(self):
		self.now = datetime.datetime.now()

	@classmethod
	def fn_seg_time(cls, sj_add_time):
		full_time = sj_add_time.split(' ')[0]
		seg_time = (int(full_time.split('-')[-1])-1)/10
		return str(seg_time) if seg_time != 3 else '2'
	
	@classmethod
	def fn_delta_time(cls, cust_add_time):
		add_time = datetime.datetime.strptime(cust_add_time, "%Y-%m-%d %H:%M:%S") if cust_add_time != '-1' else datetime.datetime.strptime("1970-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")
		return str(now - add_time).split(' day')[0] if len(str(now - add_time).split(' day')) == 2 else '0'

	@classmethod
	def fn_delta_time_2(s_time, e_time = now):
		s_time = datetime.datetime.strptime(s_time, "%Y-%m-%d %H:%M:%S")
		e_time = datetime.datetime.strptime(e_time, "%Y-%m-%d %H:%M:%S") if e_time != now else now
		return str(e_time - s_time).split(' day')[0] if len(str(e_time - s_time).split(' day')) == 2 else '0'

	@classmethod
	def fn_delta_fd_time(found_time):
		start_time = datetime.datetime.strptime(found_time, "%Y-%m-%d %H:%M:%S")
		if start_time == datetime.datetime.strptime("1970-01-01 00:00:00", "%Y-%m-%d %H:%M:%S"):
			start_time = '-1'
		if start_time != '-1':
			foundtime = str(now - start_time).split('days')[0] if len(str(now - start_time).split('days')) == 2 else '0'
		else:
			foundtime = '-1'
		return foundtime
	@classmethod
	def fn_domain_type(site_url):
		domain_1st = re.compile(r'www\.')
		domain_flg = '1' if domain_1st.findall(site_url) != [] else '0'
		return domain_flg

