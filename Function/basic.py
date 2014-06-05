from functools import partial
from multiprocessing import Pool
import math
import logging

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger('test')

def avgs(first, *rest):
	return first - sum(rest)

def anyargs(*args, **kwargs):
    print(args)      # A tuple
    print(kwargs)
    print kwargs.items()

def myfun():
	return 1,2,3

def spam(a, b, c, d):
	return a, b, c, d

#callback function
def distance(p1, p2):
	x1, y1 = p1
	x2, y2 = p2
	return math.hypot(x2 - x1, y2 - y1)

def output_result(result, log=None):
	if log is not None:
		log.debug('Got: %r', result)

def add(x, y):
	return x+y
#Carrying Extra State with Callback Functions



def main(): 
	print avgs(1,222,-100)
	anyargs('item', 'Albatross', size='large', quantity=6)
	print myfun()
	s1 = partial(spam, 1)
	print s1(2, 4, 5)
	s2 = partial(spam, d=1)
	print s2(2, 4, 5)

	points = [ (1, 2), (3, 4), (5, 6), (7, 8) ]
	pt = (4, 3)
	#points.sort(key = lambda (x,y) : -y)
	print points
	points.sort(key = lambda (x,y): distance(pt,(x,y)))
	#points.sort(key = partial(distance,pt))
	print points

	p = Pool()
	p.apply_async(add, (3, 4), callback=partial(output_result, log=log))
	p.close()
	p.join()
if __name__ == '__main__':
	main()