#%%
def latest_arr(arr, k):
    # assume K positive
    result = arr[:k]
    
    initial_sum = 0

    for indx in range(0, len(arr)-k):
        sum = 0
        for window_indx in range(k):
            sum += arr[window_indx+indx]
            
       
        result.append(sum/k)
    
    return result
# %%
data_array = [0, 1, 2, 3, 4, 5, 6, 7]
k = 2

print(latest_arr(data_array, k))
# %%
