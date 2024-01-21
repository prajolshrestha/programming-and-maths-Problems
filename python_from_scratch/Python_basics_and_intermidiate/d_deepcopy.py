import copy

original_list = [[1,2,3],
                 [4,5,6]]

copied_list = copy.deepcopy(original_list)

copied_list[0][0] = 100

print('original list', original_list)
print('copied list', copied_list)


'''
It creates a new list that is completely independent of the original list

copy.deepcopy() can be used to clone complex data structures 
like nested lists, dictionaries, objects, or any combination thereof. 
It ensures that the entire structure and its contents are duplicated, 
making it useful when you want to work with a separate copy of the data 
without altering the original.

'''