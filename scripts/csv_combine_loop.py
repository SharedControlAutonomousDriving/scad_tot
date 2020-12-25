import os
import glob
import pandas as pd

path = '/home/sai/Desktop/Marabou/scad_tot/logs/region_robustness_cexs'

path = os.path.join(path)
files = [os.path.join(path,dir,file) for dir, dir_name, file_list in os.walk(path) for file in file_list]
files_cpy = []

for filename in files:
	ls = filename.split('/')
	
	files_cpy.append(int(ls[8][6:])) 

def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)
sorted_files = argsort(files_cpy) 
files_list = [files[i] for i in sorted_files]
combined_df = pd.concat([pd.read_csv(file) for file in files_list])
combined_df.to_csv( "combined_csv.csv", index=False, encoding='utf-8-sig')