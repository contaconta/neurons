import os
import sys
import string
import csv

inRootDir  = '/raid/data/analysis/'
outRootDir = '/raid/data/analysis/'
mp4PathFileExtension = 'mp4'

for file in os.listdir(inRootDir):
	if file.endswith('.' + mp4PathFileExtension):
		file_wo_extension = file[0:(len(file) - len(ImageFileExtension) - 1)]
		file_orig = os.path.join(inRootDir, file)
		file_out  = os.path.join(inRootDir, file_wo_extension + '.avi')
		cmd_move = 'mv ' + file_orig + ' ' + file_out
		print(cmd_move)
		os.system(cmd_move) 