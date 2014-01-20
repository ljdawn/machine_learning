import multiprocessing

def worker(sign, lock):
	lock.acquire()
	print(sign, os.getpid())
	lock.release()

record = []
lock = multiprocessing.Lock()
for i in range(3):
	process = multiprocessing.Process(target=worker,args=('process',lock))
	process.start()
	record.append(process)
for process in record:
	process.join()

??multi