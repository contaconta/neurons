function processPlate(folder, resultsFolder, Sample, Identifier)



% ------------------- set the paths -----------------------
d = dir(folder);
if isempty( strfind(path, [pwd '/frangi_filter_version2a']) )
    addpath([pwd '/frangi_filter_version2a']);
end
if isempty( strfind(path, [pwd '/code']) )
    addpath([pwd '/code']);
end
if isempty( strfind(path, [pwd '/gaimc']) )
    addpath([pwd '/gaimc']);
end


% --------- generate list of folders to process -----------
count = 1;
listOfDirs = dir(folder);
for i = 1:length(listOfDirs)
    if listOfDirs(i).isdir && length(listOfDirs(i).name) > 2
        exp_num(count,:) = listOfDirs(i).name; 
        count  = count + 1;
    end
end
    
% for i = 1:240
%     exp_num(count,:) = sprintf('%03d', i); 
%     count = count + 1;
% end

filename_input = [resultsFolder 'OriginalDataDirectory.txt'];
%system(['touch ' filename_input]);
FID = fopen(filename_input, 'w');
fprintf(FID, folder);
fprintf(FID, '\n');
fprintf(FID, Sample);
fprintf(FID, '\n');
fprintf(FID, Identifier);
fclose(FID);



% ------------ process the specified folders --------------
matlabpool local

for i = 1:size(exp_num,1)
    
    tic
    folder_n = [folder exp_num(i,:) '/'];
    G = trkTracking(folder_n, resultsFolder, i , Sample);
    % perform post-processing
    a = dir([resultsFolder  sprintf('%03d', i) '.mat']);
    matFileName = a.name;
    disp(matFileName);
    if( exist([resultsFolder matFileName], 'file') > 0)
        R = load([resultsFolder matFileName]);
        R = trkPostProcessing(R, G); 
        save([resultsFolder matFileName], '-struct', 'R');
    end
    
    toc
    disp('');
    disp('=============================================================')
    disp('');
end

matlabpool close



% kill the matlab pool
%matlabpool close force



