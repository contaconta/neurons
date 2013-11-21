clear all; close all; clc;
%%
Magnification  = '10X';
datasets_paths_filename = ['PathsToDatasets-' Magnification '-20-11-2013.txt'];
outputSelectionfile = [Magnification 'StaticSelections.txt'];
OutputRootDirectory = ['/home/fbenmans/SelectionNeuriteEvaluation' Magnification '/'];
inputDataRoot      =  '/raid/data/store/';

nb_randomSelections = 100;
%%
FID = fopen([OutputRootDirectory outputSelectionfile]);
C = textscan(FID, '%s %d %d');
fclose(FID);
%%

ComplteSequencesSubDir = [OutputRootDirectory 'ComplteSequences/'];
SelectedFramesSubDir   = [OutputRootDirectory 'SelectedFrames/'];
%%

for i = 1:length(C{1})
    plateDirName = C{1}(i);
    plateDirName = plateDirName{1};
    inputDir = [plateDirName num2str(C{2}(i))];
    outputDirExp = [SelectedFramesSubDir sprintf('%03d', i) '/'];
    MP4File      = [ComplteSequencesSubDir sprintf('%03d', i) '.mp4'];
    
    RedChannelDirectory     = [inputDir '/red/'];
    Ared        = dir([RedChannelDirectory '*.TIF']);
    RedImageFileName    = [RedChannelDirectory '/' Ared(C{3}(i)).name];
    RedImageFileName = RedImageFileName(1:end-4);
    IDX = strfind(RedImageFileName, '_t');
    ImageIndex = str2num(RedImageFileName(IDX+2:end)); 
    disp('Extracting frame ')
    system(['ffmpeg -i ' MP4File ' -ss ' '0:0:' num2str((ImageIndex-1)/10)  ' ' outputDirExp '/' sprintf('%03d', i) '.jpg']);
end
