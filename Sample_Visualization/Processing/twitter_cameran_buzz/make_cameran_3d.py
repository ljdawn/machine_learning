import csv

for item in open('forpro_buzz_2.csv','rb'):
	x1, y1, x2, y2, n1, n2, t = item.strip().split(',')
	c1 = '0,135,255'
	c2 = '0,240,255'
	c3 = '0,200,252'
	co = '255,255,255'
	z1 = '200'
	z2 = '200'
	s = '2'
	s1 = '10'
	if n1 == '@ninagawamika':
		z1 = '-700'
	elif (float(x1)-595.13)*(float(x1)-595.13)+(float(y1)-494.56)*(float(y1)-494.56) < 5200 :
		z1 = '-100'
	elif (float(x1)-595.13)*(float(x1)-595.13)+(float(y1)-494.56)*(float(y1)-494.56) < 16000 :
		z1 = '0'
	elif (float(x1)-595.13)*(float(x1)-595.13)+(float(y1)-494.56)*(float(y1)-494.56) < 40000 :
		z1 = '100'
	if n2 == '@ninagawamika':
		z2 = '-700'
	elif (float(x2)-595.13)*(float(x2)-595.13)+(float(y2)-494.56)*(float(y2)-494.56) < 5200 :
		z2 = '-100'
	elif (float(x2)-595.13)*(float(x2)-595.13)+(float(y2)-494.56)*(float(y2)-494.56) < 16000 :
		z2 = '0'
	elif (float(x2)-595.13)*(float(x2)-595.13)+(float(y2)-494.56)*(float(y2)-494.56) < 40000 :
		z2 = '100'

	if n1 in '@ninagawamika':
		co = c1
		s = s1
	if n1 in ['@szgumy','@johnny_matome', '@sasakitoshinao', '@_Women_News_', '@RocketNews24', '@twinavi', '@touch_lab', '@uche_chang', '@aixca', '@otiham299', '@youpouch', '@sumito', '@internet_watch', '@hatebu', '@CINRANET', '@appbank', '@togetter_jp']: 
		co = c2
		s = s1
	if n1 in ['@HISASHI_', '@KFC_223', '@JUNO_Japan', '@ShowAyanocozey', '@ninamikainfo']:
		co = c3
		s = s1
	#if n2 in '@ninagawamika':
	#	co = c1
	#	s = s1
	#if n2 in ['@szgumy','@johnny_matome', '@sasakitoshinao', '@_Women_News_', '@RocketNews24', '@twinavi', '@touch_lab', '@uche_chang', '@aixca', '@otiham299', '@youpouch', '@sumito', '@internet_watch', '@hatebu', '@CINRANET', '@appbank', '@togetter_jp']: 
	#	co = c2
	#	s = s1
	#if n2 in ['@HISASHI_', '@KFC_223', '@JUNO_Japan', '@ShowAyanocozey', '@ninamikainfo']:
	#	co = c3
	#	s = s1


	print x1+','+y1+','+z1+','+x2+','+y2+','+z2+','+n1+','+n2+','+t+','+co+','+s




