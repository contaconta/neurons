clear all; close all; %clc
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

Nb_trkPerSeq = [];
Nb_ObjPerSequence = [];
idxSeq = 1;


for i = 1:length(listOfGTSeq)
   if(listOfGTSeq(i).isdir && ~isempty(str2num(listOfGTSeq(i).name)) ) %#ok
       currentXMLFile = [dataRootDirectory listOfGTSeq(i).name '/' listOfGTSeq(i).name '.xml'];
       disp('========================================')
       disp(listOfGTSeq(i).name)
       disp('========================================')
       AnnotatedTrackedCells = getStructureFromTrakEM2XML(dataRootDirectory, str2num(listOfGTSeq(i).name), templateHeaderFile);%#ok
       outputMatlbFile = [OutputDir listOfGTSeq(i).name];
       save(outputMatlbFile, 'AnnotatedTrackedCells');
       
       Nb_trkPerSeq(idxSeq) = numel(AnnotatedTrackedCells);%#ok
       nb_annot_obj = 0;
       for k = 1:numel(AnnotatedTrackedCells)
           nb_annot_obj = nb_annot_obj + AnnotatedTrackedCells{k}.LifeTime;
       end
       
       Nb_ObjPerSequence(idxSeq) = nb_annot_obj;%#ok
       
       idxSeq = idxSeq + 1;
   end
end