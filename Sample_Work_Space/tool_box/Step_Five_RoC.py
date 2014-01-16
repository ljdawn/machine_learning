def my_PRC(y_true, y_pred):
	from sklearn.metrics import roc_curve
	from sklearn.metrics import auc
	precision, recall, thresholds = roc_curve(y_true, y_pred)
	area = auc(precision, recall)
	return (precision, recall, thresholds, area)

if __name__ == "__main__":
	y_true = [1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1]

	y_pred = [0.43883811, 0.10542029, 0.30562706, 0.42129372, 0.62971059, 0.21760553, 0.78524454, 0.94100081, 0.89238494, 0.50591619, 0.38377724, 0.78897006, 0.91437242, 0.20083783, 0.31336396, 0.29940789, 0.58727832, 0.91174947, 0.47936179, 0.0788482 , 0.69079232, 0.30906011, 0.3327153 , 0.161681 , 0.87018128, 0.06063018, 0.34841999, 0.74680111, 0.31575738, 0.74741422, 0.67629688, 0.17078342, 0.37131395, 0.40025882, 0.0556665 , 0.06110816, 0.20600043, 0.89278627, 0.73009753, 0.87964178, 0.46559958, 0.77335994, 0.28460368, 0.12510545, 0.71503261, 0.68536381, 0.50701005, 0.08843675, 0.30114209, 0.69410203]
	print my_PRC(y_true,y_pred)[0]
	print my_PRC(y_true,y_pred)[1]
	print my_PRC(y_true,y_pred)[2]
	print my_PRC(y_true,y_pred)[3]

	import pylab as pl
	pl.clf()
	pl.plot(my_PRC(y_true,y_pred)[0], my_PRC(y_true,y_pred)[1], label='ROC curve (area = %0.2f)' % my_PRC(y_true,y_pred)[3])
	pl.plot([0, 1], [0, 1], 'k--')
	pl.xlim([0.0, 1.0])
	pl.ylim([0.0, 1.0])
	pl.xlabel('False Positive Rate')
	pl.ylabel('True Positive Rate')
	pl.title('Receiver operating characteristic example')
	pl.legend(loc="lower right")
	pl.show()