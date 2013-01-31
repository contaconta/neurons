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

% ------------------- set the paths -----------------------
if isempty( strfind(path, [pwd '/../baselFinal/frangi_filter_version2a']) )
    addpath([pwd '/../baselFinal/frangi_filter_version2a']);
end

if isempty( strfind(path, [pwd '/../baselFinal/gaimc']) )
    addpath([pwd '/../baselFinal/gaimc']);
end

if isempty( strfind(path, [pwd '/../baselFinal/Geodesics']) )
    addpath([pwd '/../baselFinal/Geodesics']);
end

if isempty( strfind(path, [pwd '/../baselFinal/ksp']) )
    addpath([pwd '/../baselFinal/ksp']);
end

if isempty( strfind(path, [pwd '/../baselFinal/fpeak']) )
    addpath([pwd '/../baselFinal/fpeak']);
end

if isempty( strfind(path, [pwd '/../baselFinal/FastEMD']) )
    addpath([pwd '/../baselFinal/FastEMD']);
end

run([pwd '/../baselFinal/vlfeat-0.9.14/toolbox/vl_setup']);

% -------- get the list of plates and process -----------
fid = fopen(datasets_paths_filename);
C = textscan(fid, '%s %s %s %s %s');
fclose(fid);

inputDataRoot      = '/raid/data/store/';
outputAnalisysRoot = '/raid/data/analysis/';

matlabpool local

for i= 1:length(C{1})
    Sample	 = C{3}(i);
    Identifier = C{4}(i);
    Location = C{5}(i);
    inputFolder = [inputDataRoot Location{1} '/original/'];
    a = dir(inputFolder);
    for j = 1:length(a)
        if( a(j).isdir && length(a(j).name) > 4 && ~isempty(regexpi(Sample{1},a(j).name)) )
            directoryName = a(j).name;
            break;
        end
    end
    inputFolder = [inputFolder   directoryName '/' ];%#ok<*AGROW>
    disp(inputFolder);
    resultsFolder = [outputAnalisysRoot  Sample{1} '/'];
    if( ~exist(resultsFolder, 'dir') )
        mkdir(resultsFolder);
    end
    processPlate(inputFolder , resultsFolder, Sample{1}, Identifier{1}, resolution);
end

matlabpool close;
