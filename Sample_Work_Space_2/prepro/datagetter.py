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
		self.reqs = """['0000220c65057869841567888ce16f46_2013-03-25','0000220c65057869841567888ce16f46_2014-05-29',]"""
		self.request = "http://10.95.28.34:8380/pgData/batchGetPgDetail?req=" + self.reqs
		self.res = []
	

	def get_response(self):
		self.response = urllib2.urlopen(self.request)
		for ans in self.response:
			self.res = json.loads(ans)['data']

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
				self.kb_times = 0
				self.kb_info = filter(lambda x:x['kb_order_flag'] in ('4','5','6'), info['kb_info'])
				self.kb_info = info['kb_info']
				self.s_pg_info = sorted(self.kb_info, key = operator.itemgetter('add_time'))
				self.sjdt_list = []
				for i in xrange(len(self.s_pg_info) - 1):
					self.dt = self.fn_delta_time_2(self.s_pg_info[i]['add_time'][:-2], self.s_pg_info[i+1]['add_time'][:-2]) if self.s_pg_info[i]['add_time'] != '' and self.s_pg_info[i+1]['add_time'] != '' else -1
					self.sjdt_list.append(dt)
				if self.s_pg_info != []:
					self.ldt = self.fn_delta_time_2(self.s_pg_info[-1]['add_time'][:-2]) if self.s_pg_info[-1]['add_time'] != '' else -1
					self.sjdt_list.append(self.ldt)
					self.sjdt_mean = np.mean(map(int ,filter(lambda x:x != '-1', self.sjdt_list)))
				for sj in self.s_pg_info:
					self.main_key = rec['key']
					self.sj_id = sj['id'] if 'id' in sj and sj['id'] != '' else '-1'
					self.cust_id = sj['cust_id'] if 'cust_id' in sj and sj['cust_id'] != '' else '-1'
					self.unit_pos_id = sj['unit_pos_id'] if 'unit_pos_id' in sj and sj['unit_pos_id'] != '' else '-1'
					self.contract_flag = sj['contract_flag'] if 'contract_flag' in sj and sj['contract_flag'] != '' else '-1'
					self.contract_flag = self.contract_flag if self.kb_times == len(self.s_pg_info)-1 else '0'
					#print '---',sj['stat']
					self.Y = 1 if sj['kb_order_flag'] == '4' else 0
					self.sj_add_time = sj['add_time'] if 'add_time' in sj and sj['add_time'] != '' else '-1'
					self.sj_end_time = sj['close_time']
					self.sj_time = self.sjdt_list[self.kb_times -1] if self.sjdt_list != [] else '-1'
					self.poi_val = poisson.pmf(int(self.sj_time), self.sjdt_mean) if self.sj_time != '-1' and self.sjdt_mean != np.nan else '-1'
					self.seg_time = self.fn_seg_time(self.sj_add_time) if self.sj_add_time != '-1' else '-1'
					self.belong_city_id = info['belong_city_id'] if 'belong_city_id' in info and info['belong_city_id'] != '' else '-1'
					self.trade_1 = info['trade_1'] if 'trade_1' in info and info['trade_1'] != '' else '-1'
					self.trade_2 = info['trade_2'] if 'trade_2' in info and info['trade_2'] != '' else '-1'
					self.type = info['type'] if 'type' in info and info['type'] != '' else '-1'
					self.site_type = info['site_type'] if 'site_type' in info and info['site_type'] != '' else '-1'
					self.no_site_type = info['no_site_type'] if 'no_site_type' in info and info['no_site_type'] != '' else '-1'
					self.hint_source_1 = info['hint_source_1'] if 'hint_source_1' in info and info['hint_source_1'] != '' else '-1'
					self.hint_source_2 = info['hint_source_2'] if 'hint_source_2' in info and info['hint_source_2'] != '' else '-1'
					self.cust_add_time = info['add_time'] if 'add_time' in info and info['add_time'] != '' else '-1'
					self.sj_delta_time = self.fn_delta_time(self.cust_add_time[:-2]) if self.cust_add_time != '-1' else '-1'
					self.registered_fund = info['registered_fund'] if 'registered_fund' in info and info['registered_fund'] != '' else '-1'
					self.found_time = info['found_time'] if 'found_time' in info and info['found_time'] != '' else '-1'
					self.delta_found_time = self.fn_delta_fd_time(self.found_time[:-2]) if self.found_time != '-1' else '-1'
					self.site_url = info['site_url'] if 'site_url' in info and info['site_url'] != [] else '-1'
					if self.site_url != '-1':site_url = ' '.join(self.site_url)
					self.domain_type = self.fn_domain_type(site_url) if self.site_url != '-1' else '-1'
					self.kb_times += 1
					self.ans = map(str,[self.main_key, self.sj_id, self.cust_id,\
						self.Y,\
						self.seg_time, self.belong_city_id, self.trade_1, self.trade_2, self.type,\
						self.site_type, self.no_site_type, self.hint_source_1, self.hint_source_2,\
						self.sj_delta_time, self.registered_fund,\
						self.delta_found_time,\
						self.domain_type,\
						self.sjdt_mean, int(self.poi_val*100000), self.kb_times])
					yield self.ans


if __name__ == '__main__':
	get_d = Datagetter()
	get_d.get_response()
	for item in get_d.get_detial():
		print '\t'.join(item)






