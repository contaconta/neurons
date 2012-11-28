function processListOfPlates(datasets_paths_filename, resolution)

% processListOfPlates from a txt file under the format provided by
% OpenBis 
% 
% This code is specific to the Sinergia project
%
% resolution could be '10x' or '20x', it's used to select parameters
%
% (c) Fethallah Benmansour, fethallah@gmail.com
%
%   Written 4/07/2012

fid = fopen(datasets_paths_filename);
C = textscan(fid, '%s %s %s %s');
fclose(fid);

inputDataRoot      = '/raid/data/store/';
outputAnalisysRoot = '/raid/data/analysis/';

for i= 1:length(C)
    Sample	 = C{1}(i);
    Identifier = C{3}(i);
    Location = C{4}(i);
    inputFolder = [inputDataRoot Location{1} '/original/'];
    a = dir(inputFolder);
    for j = 1:length(a)
        if(a(j).isdir && length(a(j).name) > 4 && length(regexpi(Identifier{1},a(j).name)) > 0)
            directoryName = a(j).name;
            break;
        end
    end
    inputFolder = [inputFolder   directoryName '/' ];%#ok<*AGROW>
    disp(inputFolder);
    resultsFolder = [outputAnalisysRoot  Identifier{1} '/'];
    if( ~exist(resultsFolder, 'dir') )
        mkdir(resultsFolder);
    end
    processPlate(inputFolder , resultsFolder, Sample{1}, Identifier{1}, resolution);
end
