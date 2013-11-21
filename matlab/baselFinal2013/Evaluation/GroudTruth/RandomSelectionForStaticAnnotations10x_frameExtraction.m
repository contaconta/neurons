clear all; close all; clc;
%%
Magnification  = '10X';
datasets_paths_filename = ['PathsToDatasets-' Magnification '-20-11-2013.txt'];
outputSelectionfile = [Magnification 'StaticSelections.txt'];
OutputRootDirectory = ['/home/fbenmans/SelectionNeuriteEvaluation' Magnification '/'];
inputDataRoot      =  '/raid/data/store/';

nb_randomSelections = 100;
%%
FID = fopen(outputSelectionfile, 'w');
for i = 1:nb_randomSelections
    Location = C{5}(PlateExpIndides(1, i));
    plateFolder = [inputDataRoot Location{1} '/original/'];
    a = dir(plateFolder);
    for j = 1:length(a)
        if(a(j).isdir && length(a(j).name) > 5)
            directoryName = a(j).name;
            break;
        end
    end
    plateFolder = [plateFolder   directoryName '/' ];%#ok<*AGROW>
    fprintf(FID, '%s \t %d %d\n', plateFolder, PlateExpIndides(2, i), PlateExpIndides(3, i));
end

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