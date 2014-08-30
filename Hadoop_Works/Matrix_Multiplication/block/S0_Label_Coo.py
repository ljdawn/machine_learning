#coding:utf8

def label_coo(data_path, h, w):
	"""matrix will be cut into m*n submatrix
		Matrix_blocks start from (1,1)
		Sub_matrix start from (0,0)	
	"""
	for lines in open(data_path,'r'):
		key_r, key_c, value = lines.strip().split(' ')
		A_r = int(key_r)/h + 1
		A_c = int(key_c)/w + 1
		print A_r, A_c, int(key_r)-(A_r-1)*h, int(key_c)-(A_c-1)*w, int(float(value)*10000)
	
if __name__ == "__main__":
	import sys
	m, n = 2, 2
	if len(sys.argv) != 2:
		print 'wrong parameter numbers'
	else:
		label_coo(sys.argv[1], m, n)

		

