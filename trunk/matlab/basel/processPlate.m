function processPlate(folder, resultsFolder, Sample)



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
for i = 1:240
    exp_num(count,:) = sprintf('%03d', i); 
    count = count + 1;
end

filename_input = [resultsFolder 'OriginalDataDirectory.txt'];
%system(['touch ' filename_input]);
FID = fopen(filename_input, 'w');
fprintf(FID, folder);
fclose(FID);



% ------------ process the specified folders --------------
for i = 1:size(exp_num,1)
    matlabpool local
    tic
    folder_n = [folder num2str(str2num(exp_num(i,:))) '/'];
    G = trkTracking(folder_n, resultsFolder, i , Sample);
    % perform post-processing
    a = dir([resultsFolder '*'  num2str(str2num(exp_num(i,:))) '.mat']);
    matFileName = a.name;
    disp(matFileName);
    if( exist([resultsFolder matFileName], 'file') > 0)
        R = load([resultsFolder matFileName]);
        R = trkPostProcessing(R, G); 
        save([resultsFolder matFileName], '-struct', 'R');
    end
    
    toc
    matlabpool close
    disp('');
    disp('=============================================================')
    disp('');
end



% kill the matlab pool
matlabpool close force



