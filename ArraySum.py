def sum(array):
    x = 0
    if(array != []):
        x+=array[0]
        sum(array[1:])
    else:
        return(x)

arr = [1,1,1,1,1]
print(sum(arr))