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

print_red_on_white = lambda x: cprint(x, 'red', 'on_white')
print_green_on_white = lambda x: cprint(x, 'green', 'on_white')
print_blue_on_white = lambda x: cprint(x, 'blue', 'on_white')
print_white_on_blue = lambda x: cprint(x, 'white', 'on_blue')
print_white_on_red = lambda x: cprint(x, 'white', 'on_red')
print_white_on_green = lambda x: cprint(x, 'white', 'on_green')

AnalysisDir = '/raid/data/analysis/'

ExtenstionsList = ['.jpg', 'params.mat', '.webm', '.mp4', '.mat']
numberOfProcessedPlates = 0
numberOfCompleteProcessedPlates = 0
numberOfUncompleteProcessedPlates = 0
numberOfUnprocessedPlates = 0

ListOfCompletePlates = []

ListOfNonCompletePlates = []
ListOfSizeNonCompletePlates = []

ListOfNonProcessedPlates = []

ShowSummaryOnly = 1

for plateName in os.listdir(AnalysisDir):
	PlateDir = os.path.join(AnalysisDir, plateName)
	if os.path.isdir(PlateDir):
		if ShowSummaryOnly != 1:
			print colored(plateName, 'cyan')
			
		platePropertiesFile = os.path.join(PlateDir, 'OriginalDataDirectory.txt');
		if os.path.exists(platePropertiesFile):
			if ShowSummaryOnly != 1:
				os.system('cat ' + platePropertiesFile)
				print '\n'
			numberOfProcessedPlates = numberOfProcessedPlates + 1
			for extension in ExtenstionsList:
				kkk = os.path.join(PlateDir, '*' + extension)
				cmd_ = 'ls ' + kkk + ' | wc -w' 
				numberOfFilesExt = GetFloatingPointNumberFromExecOutput(cmd_)
				if numberOfFilesExt[0] == 240:
					if ShowSummaryOnly != 1:
						print colored(str(numberOfFilesExt) +' ' + extension + ' files', 'green')
					if extension == '.jpg':
						numberOfCompleteProcessedPlates = numberOfCompleteProcessedPlates + 1
						ListOfCompletePlates.append(plateName)
				elif numberOfFilesExt[0] > 0:
					if ShowSummaryOnly != 1:
						print colored(str(numberOfFilesExt) +' ' + extension + ' files', 'blue')
					if extension == '.jpg':
						numberOfUncompleteProcessedPlates = numberOfUncompleteProcessedPlates + 1
						ListOfNonCompletePlates.append(plateName)
						ListOfSizeNonCompletePlates.append(numberOfFilesExt[0])
				else:
					if ShowSummaryOnly != 1:
						print colored(str(numberOfFilesExt) +' ' + extension + ' files', 'red')
					if extension == '.jpg':
						numberOfUnprocessedPlates = numberOfUnprocessedPlates + 1
						ListOfNonProcessedPlates.append(plateName)
			if ShowSummaryOnly != 1:
				print_red_on_white( '---------------------------------------------------------------')
				print '\n'


print_red_on_white('---------------------------------------------------------------')
print_red_on_white('-------------------------  Summary  ---------------------------')
print_red_on_white('---------------------------------------------------------------')
print_green_on_white('Total number of plates                          :            ' + str( numberOfProcessedPlates ))
print_green_on_white('Number of plates with 240 stages                :            ' + str( numberOfCompleteProcessedPlates ))
for a in ListOfCompletePlates:
	print_white_on_green(a + 'complete !! ' )
print_blue_on_white( 'Number of plates with less than 240 stages      :             ' + str( numberOfUncompleteProcessedPlates ))
i = 0
for a in ListOfNonCompletePlates:
	print_white_on_blue(str(ListOfSizeNonCompletePlates[i]) + ' stages in ' + a )
	i = i+1

print_red_on_white(  'Number of plates not processed ( no red channel):             ' + str( numberOfUnprocessedPlates ))
for a in ListOfNonProcessedPlates:
	print_white_on_red('No red channels for this plate : ' + a )

