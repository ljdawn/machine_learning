#cording:utf-8

import datetime
import urllib2
import cPickle as pickle
import json
import re
import operator
import numpy as np
from scipy.stats import poisson

now = datetime.datetime.now()

class Datagetter(object):
	"""docstring for Datagetter"""
	def __init__(self):
		#self.reqs = str(part_keys).replace("u'","'").replace(", ",",")
		self.reqs = """['0000220c65057869841567888ce16f46_2013-03-25','0000220c65057869841567888ce16f46_2014-05-29','000035ff8f6b8e99cb1d1edf6178172f_2013-03-12','0000becc4f5b0427d89950f3fdbfaa8a_2014-06-20','0000e26928c40934a41af0e37fb9aa9c_2014-04-02']"""
		self.request = "http://10.95.28.34:8380/pgData/batchGetPgDetail?req=" + self.reqs
		self.res = []
	
	def get_response(self):
		self.response = urllib2.urlopen(self.request)
		for ans in self.response:
			self.res = json.loads(ans)['data']

	def get_value(self, dic, key, def_value = '-1'):
		return dic[key] if key in dic and dic[key] not in ('', [])  else def_value

	def get_fn_value(self, fn, x):
		return apply(fn, x) if x not in ('-1', []) else '-1'

	def get_data(self):
		return self.res

	def save_data(self, data_path):
		with open(data_path, 'wb') as pdfile:
			for rec in self.get_detial():
				line = ','.join(rec)
				pdfile.write(line+'\n')

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
	def fn_delta_time_2(cls, s_time, e_time = now):
		s_time = datetime.datetime.strptime(s_time, "%Y-%m-%d %H:%M:%S")
		e_time = datetime.datetime.strptime(e_time, "%Y-%m-%d %H:%M:%S") if e_time != now else now
		return str(e_time - s_time).split(' day')[0] if len(str(e_time - s_time).split(' day')) == 2 else '0'

	@classmethod
	def fn_delta_fd_time(cls, found_time):
		start_time = datetime.datetime.strptime(found_time, "%Y-%m-%d %H:%M:%S")
		if start_time == datetime.datetime.strptime("1970-01-01 00:00:00", "%Y-%m-%d %H:%M:%S"):
			start_time = '-1'
		if start_time != '-1':
			foundtime = str(now - start_time).split('days')[0] if len(str(now - start_time).split('days')) == 2 else '0'
		else:
			foundtime = '-1'
		return foundtime
	
	@classmethod
	def fn_domain_type(cls, site_url):
		domain_1st = re.compile(r'www\.')
		domain_flg = '1' if domain_1st.findall(site_url) != [] else '0'
		return domain_flg

	def get_detial(self):
		for rec in self.res:
			for info in rec['pgInfos']:
				kb_times = 0
				self.kb_info = filter(lambda x:x['kb_order_flag'] in ('4','5','6'), info['kb_info'])
				self.s_pg_info = sorted(self.kb_info, key = operator.itemgetter('add_time'))
				self.sjdt_list = []
				for i in xrange(len(self.s_pg_info) - 1):
					self.dt = self.fn_delta_time_2(self.s_pg_info[i]['add_time'][:-2], self.s_pg_info[i+1]['add_time'][:-2]) if self.s_pg_info[i]['add_time'] != '' and self.s_pg_info[i+1]['add_time'] != '' else -1
					self.sjdt_list.append(self.dt)
				if self.s_pg_info != []:
					self.ldt = self.fn_delta_time_2(self.s_pg_info[-1]['add_time'][:-2]) if self.s_pg_info[-1]['add_time'] != '' else -1
					self.sjdt_list.append(self.ldt)
					self.sjdt_mean = np.mean(map(int, filter(lambda x:x != '-1', self.sjdt_list)))
				for sj in self.s_pg_info:
					main_key = rec['key']
					sj_labels = ('id', 'cust_id', 'unit_pos_id', 'contract_flag','add_time')
					sj_dict = (sj,)*len(sj_labels)
					info_labels = ('belong_city_id', 'trade_1', 'trade_2', 'type', 'site_type', 'no_site_type', 'hint_source_1', 'hint_source_2', 'add_time', 'registered_fund', 'found_time', 'site_url')
					info_dict = (info,)*len(info_labels)
					(belong_city_id, trade_1, trade_2, info_type, site_type, no_site_type, hint_source_1, hint_source_2, cust_add_time, registered_fund, found_time, site_url)=map(self.get_value, info_dict, info_labels)
					(sj_id, cust_id, unit_pos_id, contract_flag, sj_add_time) = map(self.get_value, sj_dict, sj_labels)
					Y = 1 if sj['kb_order_flag'] == '4' else 0
					sj_time = self.sjdt_list[kb_times -1] if self.sjdt_list != [] else '-1'
					poi_val = poisson.pmf(int(sj_time), self.sjdt_mean) if sj_time != '-1' and self.sjdt_mean != np.nan else '-1'
					seg_time = self.fn_seg_time(sj_add_time) if sj_add_time != '-1' else '-1'	
					sj_delta_time = self.fn_delta_time(cust_add_time[:-2]) if cust_add_time != '-1' else '-1'
					delta_found_time = self.fn_delta_fd_time(found_time[:-2]) if found_time != '-1' else '-1'
					if site_url != '-1':site_url = ' '.join(site_url)
					domain_type = self.get_fn_value(self.fn_domain_type, site_url)
					turn_out_count = len(sj['turn_out_infos'])
					turn_out_time_mean = '-1'
					if turn_out_count != 0:
						
					kb_times += 1
					self.ans = map(str,[main_key, sj_id, cust_id, Y, seg_time, belong_city_id, trade_1, trade_2, info_type, site_type, \
						no_site_type, hint_source_1, hint_source_2, sj_delta_time, registered_fund, delta_found_time, domain_type, self.sjdt_mean, \
						int(poi_val*100000), kb_times, turn_out_count])
					yield self.ans

if __name__ == '__main__':
	get_d = Datagetter()
	get_d.get_response()
	for item in get_d.get_detial():
		print '\t'.join(item)





