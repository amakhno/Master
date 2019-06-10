import pandas as pd
import numpy as np
import json
import os

def compare_h5(fn1, fn2):
    store1 = pd.HDFStore(fn1)
    store2 = pd.HDFStore(fn2)    

    y1 = pd.DataFrame(store1.root.Y)
    y2 = pd.DataFrame(store2.root.Y)
    x1_shape, _ = y1.shape
    y1_final_abundances = y1.iloc[[x1_shape - 1]]
    y2_final_abundances = y2.iloc[[x1_shape - 1]]

    result = []
    
    for index in range(y1_final_abundances.size):
        a = store1.root.A[index]
        z = store1.root.Z[index]
        current_y1 = y1_final_abundances.iat[0, index]
        current_y2 = y2_final_abundances.iat[0, index]
        if (current_y1 != current_y2):
            print("Hi")
        dy = abs(current_y1 - current_y2)
        if (current_y1 == 0.0):
            delta_y = 0
        else:
            delta_y = dy/current_y1
        result.append([
            a, 
            z,
            current_y1, 
            current_y2,
            dy,
            delta_y
        ])

    store1.close()
    store2.close()
    return result

def write_out_compare():
    folder_name = 'out-r-full'
    for index in range(50, 56):
        # print(folder_name + '/WithMy-' + str(index))
        result = compare_h5(folder_name + '/WithMy-' + str(index) + '.h5', 
            folder_name + '/WithoutMy-' + str(index) + '.h5')
        #with open('out-compare/data-{0}.json'.format(index), 'w') as outfile:
            ##json.dump(result, outfile)
        #print("multyplier: 1e{0}, diff: {1}".format(index, diff))

def print_compare_result():
    for index in range(32, 56):
        # print(folder_name + '/WithMy-' + str(index))
        filename = 'out-compare/data-{0}.json'.format(index)
        print(filename)
        if (os.path.isfile(filename)):
            with open(filename, 'r') as infile:            
                content = infile.read()
                result = json.loads(content)
        else:
            continue

        #dataFrame = pd.DataFrame(result)
        #dataFrame.rename(columns = {'A', 'Z', 'Y1', 'Y1', 'DY', 'DeltaY'}, inplace=True)
        for arr_index in range(len(result)):
            if (result[arr_index][5] != 0.0):
                raise "Hello"
            #print(result[arr_index])            
        #print("multyplier: 1e{0}, diff: {1}".format(index, diff))

#write_out_compare()

print_compare_result()