def bubble_sort(List):
	for j in xrange(len(List)-1, 0, -1):
		for i in xrange(0,j):
			if List[i] > List[i+1]:List[i], List[i+1] = List[i+1], List[i]
	return List

def selection_sort(List):
	for i in xrange(len(List)-1):
		min_index = i
		for j in xrange(i+1, len(List)):
			if List[min_index] > List[j]:
				min_index = j
	return List

def insertion_sort(List):
	right_l = List[1:]
	for i in xrange(len(right_l)):
		if List[0] <= right_l[i]:
			List = right_l[:i] + [List[0]] + right_l[i:]
	return [List[0]] + right_l

def quick_sort(arr):
	if len(arr) < 1:
		return arr
	key = arr[0]
	left = []
	right = []
	for i in xrange(1, len(arr)):
		if arr[i] <= key:
			left.append(arr[i])
		else:
			right.append(arr[i])
	left = quick_sort(left)
	right = quick_sort(right)
	return left + [key] + right

		

if __name__ == '__main__':
	test_list = [12,3,51,3,2,5,78,88,34,25,2,1000]
	#print bubble_sort(test_list)
	#print selection_sort(test_list)
	#print insertion_sort(test_list)
	#print quick_sort(test_list)