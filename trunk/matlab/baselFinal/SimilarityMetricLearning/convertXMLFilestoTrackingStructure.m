clear all; close all; clc
%%
templateHeaderFile  = '../Datasets/TemplateHeaderSimplified.xml';
addpath('../Datasets/')

Magnification = '10x';
dataRootDirectory = ['/Users/feth/Google Drive/Sinergia/GT' Magnification '/Dynamic/'];
OutputDir         = ['/Users/feth/Google Drive/Sinergia/GT' Magnification '/Dynamic_matlab/'];
%%
if(~exist(OutputDir, 'dir'))
    mkdir(OutputDir);
end
%%
listOfGTSeq = dir(dataRootDirectory);

for i = 5:5%length(listOfGTSeq)
   if(listOfGTSeq(i).isdir && ~isempty(str2num(listOfGTSeq(i).name)) ) %#ok
       currentXMLFile = [dataRootDirectory listOfGTSeq(i).name '/' listOfGTSeq(i).name '.xml'];
       disp('========================================')
       disp(listOfGTSeq(i).name)
       disp('========================================')
       AnnotatedTrackedCells = getStructureFromTrakEM2XML(dataRootDirectory, str2num(listOfGTSeq(i).name), templateHeaderFile);%#ok
       outputMatlbFile = [OutputDir listOfGTSeq(i).name];
       save(outputMatlbFile, 'AnnotatedTrackedCells');
   end
end