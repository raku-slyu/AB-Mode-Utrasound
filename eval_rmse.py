import os
import numpy as np

path = './results'
files = os.listdir(path)

for cur_file in files:
	if 'csv' not in cur_file:
		continue
	if 'length' not in cur_file:
		continue
	cur_file_path = os.path.join(path, cur_file)
	y= np.loadtxt(cur_file_path, delimiter=',')
	gt = y[:,0]
	pred = y[:,1]
	print(cur_file)
	print(np.mean((gt-pred)**2)**.5)
