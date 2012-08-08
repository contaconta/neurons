function processPlateMSER(folder, resultsFolder, Sample, Identifier)


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
if isempty( strfind(path, [pwd '/../basel/frangi_filter_version2a']) )
    addpath([pwd '/../basel/frangi_filter_version2a']);
end
if isempty( strfind(path, [pwd '/../basel/code']) )
    addpath([pwd '/../basel/code']);
end
if isempty( strfind(path, [pwd '/../basel/gaimc']) )
    addpath([pwd '/../basel/gaimc']);
end

addpath('/home/fbenmans/src/neurons/matlab/basel10x/RegionGrowingSomata/');
addpath('/home/fbenmans/src/WLV/main/');
addpath('/home/fbenmans/src/WLV/matlab/');
run('~/Downloads/vlfeat-0.9.14/toolbox/vl_setup');
addpath(genpath('~/Downloads/MatlabFns/'));

% --------- generate list of folders to process -----------
count = 1;
listOfDirs = dir(folder);
for i = 1:length(listOfDirs)
    if listOfDirs(i).isdir && length(listOfDirs(i).name) > 2
        exp_num(count,:) = listOfDirs(i).name; %#ok<*AGROW>
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
for i = 9:size(exp_num,1)
    
    tic
    folder_n = [folder exp_num(i,:) '/'];
    G = trkTrackingMSER(folder_n, resultsFolder, exp_num(i,:), Sample);
    a = dir([resultsFolder  exp_num(i,:) '.mat']);
%     matFileName = a.name;
%     disp(matFileName);
%     if( exist([resultsFolder matFileName], 'file') > 0)
%         R = load([resultsFolder matFileName]);
%         R = trkPostProcessing(R, G); %#ok
%         save([resultsFolder matFileName], '-struct', 'R');
%     end
    
    toc
    disp('');
    disp('=============================================================')
    disp('');
end

%matlabpool close



% kill the matlab pool
%matlabpool close force



