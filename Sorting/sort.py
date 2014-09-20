#coding:utf-8

#两两比价，一轮下来最大值排在尾部，下一轮排序排数组［－1］个元素，依此类推。
def bubble_sort(List):
	for j in xrange(len(List)-1, 0, -1):
		for i in xrange(0,j):
			if List[i] > List[i+1]:List[i], List[i+1] = List[i+1], List[i]
	return List

#两两比价，最小放头部，下一轮排序排数组［＋1：］个元素，依此类推。但是每轮直交换1回，较快。
def selection_sort(List):
	for i in xrange(len(List)-1):
		min_index = i
		for j in xrange(i+1, len(List)):
			if List[min_index] > List[j]:
				min_index = j
		List[i], List[min_index] = List[min_index], List[i]
	return List


#找一个基准值，通常是数组第一个元素，小于它的排左边，大于它的排右边，它排在中间，返回，递归。python程序没有交换过程。
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

#分解数组，分解到最小单位。
def merge_sort(arr):
	if len(arr) <= 1:
		return arr
	mid = len(arr) / 2
	left = arr[:mid]
	right = arr[mid:]

	left = merge_sort(left)
	right = merge_sort(right)

	return merge(left, right)

#merge,左右两个数组，从头比较，从小的开始加入结果，剩下的也加进来，形成一次merge, 两个数组需要都是排好序的，合并的数组才是排好顺序的。
def merge(left, right):
	res = []
	left_index, right_index = 0, 0
	while left_index < len(left) and right_index < len(right):
		if left[left_index] <= right[right_index]:
			res.append(left[left_index])
			left_index += 1
		else:
			res.append(right[right_index])
			right_index += 1
	if left:
		res.extend(left[left_index:])
	if right:
		res.extend(right[right_index:])
	return res

def binary_search(l, value):
	low = 0
	high = len(l) - 1
	while low <= high:
		mid = (low + high) / 2
		if l[mid] == value:
			return mid
		elif l[mid] < value:
			low = mid + 1
		else:
			high = mid - 1
	return 'not found!'


if __name__ == '__main__':
	test_list = [12,3,51,3,2,5,78,88,34,25,2,1000]
	#print bubble_sort(test_list)
	#print selection_sort(test_list)
	#print quick_sort(test_list)
	#print merge(test_list[:5], test_list[5:])
	#print merge_sort(test_list)
	sorted_list = merge_sort(test_list)
	print sorted_list
	print binary_search(sorted_list, 10)
