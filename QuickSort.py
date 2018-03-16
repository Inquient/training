def qsort(arr):
   if len(arr)<2:
       return arr
   else:
       pivot = arr[0]
       less = [i for i in arr[1:] if i<=pivot]
       greater = [i for i in arr[1:] if i>pivot]
   return qsort(less)+[pivot]+qsort(greater)

a = [4,14,3,2,7,8,0,3,2,6]
print(qsort(a))