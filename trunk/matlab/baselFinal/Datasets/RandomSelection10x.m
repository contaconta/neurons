clear all; close all; clc;
%%
Magnification  = '10X';
datasets_paths_filename = ['PathsToDatasets-' Magnification '-28112012.txt'];
outputSelectionfile = [Magnification 'Selections.txt'];
OutputRootDirectory = ['/home/fbenmans/Selection' Magnification '/'];
inputDataRoot      =  '/raid/data/store/';

nb_randomSelections = 50;
%%
FID = fopen(datasets_paths_filename);
C = textscan(FID, '%s %s %s %s %s');
fclose(FID);
%%
NumOfSequences = zeros(1, length(C{1}));

for i= 1:length(C{1})
    Sample	 = C{3}(i);
    Location = C{5}(i);
    
    plateFolder = [inputDataRoot Location{1} '/original/'];
    a = dir(plateFolder);
    for j = 1:length(a)
        if(a(j).isdir && length(a(j).name) > 5)
            directoryName = a(j).name;
            break;
        end
    end
    
    plateFolder = [plateFolder   directoryName '/' ];%#ok<*AGROW>
    a = dir(plateFolder);
    for j = 1:length(a)
        if(a(j).isdir)
            NumOfSequences(i)  = NumOfSequences(i)  +1;
        end
    end
    NumOfSequences(i)  = NumOfSequences(i) - 2;
    if strcmp(Magnification, '20x')
        if i <=4
            NumOfSequences(i)  = NumOfSequences(i) - 1;
        end
    end
    
end
%%
TotalNumberOfSequences = sum(NumOfSequences);

randomPlateSelection = randi(nb_randomSelections, [1, nb_randomSelections]);
PlateExpIndides = zeros(2, nb_randomSelections);

for i = 1:nb_randomSelections
    PlateExpIndides(1, i) = randomPlateSelection(i);
    PlateExpIndides(2, i) = randi(NumOfSequences(PlateExpIndides(1, i)), 1);
end
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
    fprintf(FID, '%s \t %d \n', plateFolder, PlateExpIndides(2, i));
end

fclose(FID);
%%
if ~exist(OutputRootDirectory, 'dir')
    mkdir(OutputRootDirectory);
end
system(['cp ' outputSelectionfile ' ' OutputRootDirectory]);

FID = fopen([OutputRootDirectory outputSelectionfile]);
C = textscan(FID, '%s %d');
fclose(FID);

for i = 1:length(C{1})
    plateDirName = C{1}(i);
    plateDirName = plateDirName{1};
    inputDir = [plateDirName num2str(C{2}(i))];
    outputDirExp = [OutputRootDirectory sprintf('%03d', i) '/'];
    
    if  exist(outputDirExp, 'dir')
        rmdir(outputDirExp, 's');
    else
        mkdir(outputDirExp);
    end
    
    copy_cmd = ['cp -r ' inputDir '/red/ ' outputDirExp '/red/'];
    disp(copy_cmd);
    system(copy_cmd);
    copy_cmd = ['cp -r ' inputDir '/green/ ' outputDirExp '/green/'];
    system(copy_cmd);
end