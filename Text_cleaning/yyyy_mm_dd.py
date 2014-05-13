#coding:utf8
import re

def find_datatime_in_string(text):
	rule_1 = '(1[8-9]\d{2}|2[0-1]\d{2})([-\/\._ ])(10|12|0?[13578])([-\/\._ ])(3[01]|[12][0-9]|0?[1-9])$'
	rule_2 = '(1[8-9]\d{2}|2[0-1]\d{2})([-\/\._ ])(11|0?[469])([-\/\._ ])(30|[12][0-9]|0?[1-9])$'
	rule_3 = '(1[8-9]\d{2}|2[0-1]\d{2})([-\/\._ ])(0?2)([-\/\._ ])(2[0-8]|1[0-9]|0?[1-9])$'
	rule_4 = '(2000)([-\/\._ ])(0?2)([-\/\._ ])(29)' 
	lyear = range(1800, 2100)
	leap_year = []
	for year in lyear:
		if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
			leap_year.append(str(year))
	rule_5_h = '|'.join(leap_year)
	rule_5 = 'rule_5_h'+'([-\/\._ ])(0?2)([-\/\._ ])(29)$'
	rule = re.compile(rule_1+'|'+rule_2+'|'+rule_3+'|'+rule_4)
	rule_0 = re.compile('\d{4}[-\/\._ ]\d{2}[-\/\._ ]\d{2}')
	ans = []
	for st in rule_0.findall(text):
		try:
			ans.append(rule.match(st).group())
		except:
			pass
	return ans if ans != [] else ['nothing']

def main():
	text = 'zhognguo工商局 2651 ewf 1985-89-98 fafe//,dlmcv a2014 02_29 ae f a 2000-02-291922 02-02'
	text = 'afwef 1922-12-32  1981 01 01 www.baidu.com ]]'
	print 'input text :', text
	print 'we have found datatime :',

	print ' '.join([datatime for datatime in find_datatime_in_string(text)])

if __name__ == '__main__':
	main()