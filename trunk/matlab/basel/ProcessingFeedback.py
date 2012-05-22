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
import colored
from termcolor import colored

sys.path.append("./")
from GetFloatingPointNumberFromExecOutput import *

AnalysisDir = '/raid/data/analysis/'

ExtenstionsList = ['.jpg', 'params.mat', '.webm', '.mp4', '.mat']


for plateName in os.listdir(AnalysisDir):
	PlateDir = os.path.join(AnalysisDir, plateName)
	if os.path.isdir(PlateDir):
		print colored('plateName', 'magenta')
		print '\n'
		platePropertiesFile = os.path.join(PlateDir, 'OriginalDataDirectory.txt');
		if os.path.exists(platePropertiesFile):
			os.system('cat ' + platePropertiesFile)
			print '\n'
		for extension in ExtenstionsList
			kkk = os.path.join(PlateDir, '*' + extension)
			cmd_ = 'ls ' + kkk + ' | wc -w' 
			numberOfFilesExt = GetFloatingPointNumberFromExecOutput(cmd_)
			if numberOfFilesExt == 240
				print colored(str(numberOfFilesExt) +' ' + extension + ' files', 'green')
			elif  numberOfFilesExt > 0 
				print colored(str(numberOfFilesExt) +' ' + extension + ' files', 'blue')
			else	
				print colored(str(numberOfFilesExt) +' ' + extension + ' files', 'red')
		print 'done!!'




