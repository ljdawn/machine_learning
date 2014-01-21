import time

def timer(f):
	def func(*args, **kwargs):
		start = time.time()
		res = f(*args, **kwargs)
		end  = time.time()
		print f.__name__, 'took', end - start, 'time'
		return res
	return func

if __name__ == '__main__':
	@timer
	def test(word ='hello'):
		print word
	test()