#//////////////////////////////////////////////////////////////////////////////////
#//																																							 //
#// Copyright (C) 2012 Fethallah Benmansour																			 //
#//																																							 //
#// This program is free software: you can redistribute it and/or modify         //
#// it under the terms of the version 3 of the GNU General Public License        //
#// as published by the Free Software Foundation.                                //
#//                                                                              //
#// This program is distributed in the hope that it will be useful, but          //
#// WITHOUT ANY WARRANTY; without even the implied warranty of                   //
#// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU             //
#// General Public License for more details.                                     //
#//                                                                              //
#// You should have received a copy of the GNU General Public License            //
#// along with this program. If not, see <http://www.gnu.org/licenses/>.         //
#//                                                                              //
#// Contact <fethallah@gmail.com> for comments & bug reports										 //
#//////////////////////////////////////////////////////////////////////////////////

import os
import sys

sys.path.append("./")
sys.path.append("termcolor")
from GetFloatingPointNumberFromExecOutput import *
from termcolor import colored, cprint

print_red_on_cyan = lambda x: cprint(x, 'red', 'on_cyan')
print_green_on_cyan = lambda x: cprint(x, 'green', 'on_cyan')
print_blue_on_cyan = lambda x: cprint(x, 'blue', 'on_cyan')


AnalysisDir = '/raid/data/analysis/'

ExtenstionsList = ['.jpg', 'params.mat', '.webm', '.mp4', '.mat']
numberOfProcessedPlates = 0
numberOfCompleteProcessedPlates = 0
numberOfUncompleteProcessedPlates = 0
numberOfUnprocessedPlates = 0

for plateName in os.listdir(AnalysisDir):
	PlateDir = os.path.join(AnalysisDir, plateName)
	if os.path.isdir(PlateDir):
		print colored(plateName, 'cyan')
		platePropertiesFile = os.path.join(PlateDir, 'OriginalDataDirectory.txt');
		if os.path.exists(platePropertiesFile):
			os.system('cat ' + platePropertiesFile)
			print '\n'
			numberOfProcessedPlates = numberOfProcessedPlates + 1
			for extension in ExtenstionsList:
				kkk = os.path.join(PlateDir, '*' + extension)
				cmd_ = 'ls ' + kkk + ' | wc -w' 
				numberOfFilesExt = GetFloatingPointNumberFromExecOutput(cmd_)
				if numberOfFilesExt[0] == 240:
					print colored(str(numberOfFilesExt) +' ' + extension + ' files', 'green')
					if extension == 'jpg':
						numberOfCompleteProcessedPlates = numberOfCompleteProcessedPlates + 1
				elif numberOfFilesExt[0] > 0:
					print colored(str(numberOfFilesExt) +' ' + extension + ' files', 'blue')
					if extension == 'jpg':
						numberOfUncompleteProcessedPlates = numberOfUncompleteProcessedPlates + 1
				else:
					print colored(str(numberOfFilesExt) +' ' + extension + ' files', 'red')
					if extension == 'jpg':
						numberOfUnprocessedPlates = numberOfUnprocessedPlates + 1
			print '---------------------------------------------------------------'
			print '\n'


print_red_on_cyan('---------------------------------------------------------------')
print_red_on_cyan('Summary')
print_red_on_cyan('---------------------------------------------------------------')
print_green_on_cyan('Total number of plates      : ' + str( numberOfProcessedPlates ))
print_green_on_cyan('Number of plates with 240   : ' + str( numberOfCompleteProcessedPlates ))
print_blue_on_cyan('Number of plates with < 240 : ' + str( numberOfProcessedPlates ))
print_red_on_cyan('Number of plates not processed (probably no red channel): ' + str( numberOfUnprocessedPlates ))
