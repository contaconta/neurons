clear all; close all; clc;
%%
addpath('../');
Magnification = '20x';
dataRootDirectory    = ['/Users/feth/Google Drive/Sinergia/GT' Magnification '/Dynamic/'];
ConvertedGTRootDir   = ['/Users/feth/Google Drive/Sinergia/GT' Magnification '/Dynamic_matlab/'];
RawRootDataDirectory = ['/Users/feth/Documents/Work/Data/Sinergia/Olivier/Selection' Magnification '/'];
%%
listOfGTSeq = dir(dataRootDirectory);

for i = 1:length(listOfGTSeq)
   if(listOfGTSeq(i).isdir && ~isempty(str2num(listOfGTSeq(i).name)) ) %#ok
       inputSeqDirToProcess = [RawRootDataDirectory listOfGTSeq(i).name '/'];
       disp('========================================')
       disp(inputSeqDirToProcess);
       disp('========================================')
       [Nuclei, Somata] =  trkDetectNucleiSomataForEvaluation(inputSeqDirToProcess, Magnification);
       keyboard;
   end
end