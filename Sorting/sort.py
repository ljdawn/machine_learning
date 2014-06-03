def bubble_sort(List):
	for j in xrange(len(List)-1, 0, -1):
		for i in range(0,j):
			if List[i] > List[i+1]:List[i], List[i+1] = List[i+1], List[i]
	return List

test_list = [2,3,51,3,2,5,78,88,34,25]

print bubble_sort(test_list)