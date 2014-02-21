import csv

for item in open('forpro_buzz_2.csv','rb'):
	x1, y1, x2, y2, n1, n2, t = item.strip().split(',')
	z1 = '0'
	z2 = '0'
	if n1 in '@ninagawamika':
		z1 = '300'
	if n1 in ['@szgumy','@johnny_matome', '@sasakitoshinao', '@_Women_News_', '@RocketNews24', '@twinavi', '@touch_lab', '@uche_chang', '@aixca', '@otiham299', '@youpouch', '@sumito', '@internet_watch', '@hatebu', '@CINRANET', '@appbank', '@togetter_jp']: 
		z1 = '100'
	if n1 in ['@HISASHI_', '@KFC_223', '@JUNO_Japan', '@ShowAyanocozey', '@ninamikainfo']:
		z1 = '250'
	if n2 in '@ninagawamika':
		z2 = '300'
	if n2 in ['@szgumy','@johnny_matome', '@sasakitoshinao', '@_Women_News_', '@RocketNews24', '@twinavi', '@touch_lab', '@uche_chang', '@aixca', '@otiham299', '@youpouch', '@sumito', '@internet_watch', '@hatebu', '@CINRANET', '@appbank', '@togetter_jp']: 
		z2 = '100'
	if n2 in ['@HISASHI_', '@KFC_223', '@JUNO_Japan', '@ShowAyanocozey', '@ninamikainfo']:
		z2 = '250'
	print x1+','+y1+','+z1+','+x2+','+y2+','+z2+','+n1+','+n2+','+t
