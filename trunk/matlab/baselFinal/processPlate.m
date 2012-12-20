function processPlate(folder, resultsFolder, Sample, Identifier, magnification)


% processPlate: process one plate containing sub-directories
% 
% folder        : is the input folder
% resultsFolder : the output folder where to write the results
% Sample        : a string describing the experiment (or the plate, ex: PLATE3-G4)
% Identifier    : a string Identifier for the experiment
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

run([pwd '/../baselFinal/vlfeat-0.9.14/toolbox/vl_setup']);

% --------- generate list of folders to process -----------
count = 1;
listOfDirs = dir(folder);
for i = 1:length(listOfDirs)
    if listOfDirs(i).isdir && ~isempty(str2num(listOfDirs(i).name))%#ok
        exp_num{count} = listOfDirs(i).name; %#ok<*AGROW>
        count  = count + 1;
    end
end
    

filename_input = [resultsFolder 'OriginalDataDirectory.txt'];
FID = fopen(filename_input, 'w');
fprintf(FID, folder);
fprintf(FID, '\n');
fprintf(FID, Sample);
fprintf(FID, '\n');
fprintf(FID, Identifier);
fclose(FID);



% ------------ process the specified folders --------------
% matlabpool local
for i = 1:length(exp_num)
    
    folder_n = [folder exp_num{i} '/'];
    initime = cputime;
    trkTracking(folder_n, resultsFolder, exp_num{i}, Sample, magnification);
    endtime = cputime;
    fprintf('CPUTIME: %g \n', endtime-initime);
    disp('');
    disp('=============================================================')
    disp('');
end

%matlabpool close



% kill the matlab pool
%matlabpool close force



