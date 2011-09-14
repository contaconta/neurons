
%% ICFILER
folder = '/net/cvlabfiler1/home/ksmith/Basel/ControlScreen/Plate1_10-5-2010/';
resultsFolder = '/net/cvlabfiler1/home/ksmith/Basel/ControlScreen/Results/';

%% kevin's laptop
%folder = '/home/ksmith/data/basel/ControlScreen/Plate1_10-5-2010/';
%resultsFolder = '/home/ksmith/data/basel/ControlScreen/Results/Plate1/';



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
for i = 4:4  %1:240
    exp_num(count,:) = sprintf('%03d', i); %#ok<SAGROW>
    count = count + 1;
end


% ------------ process the specified folders --------------
for i = 1:size(exp_num,1)
    matlabpool local
    tic
    folder_n = [folder exp_num(i,:) '/'];
    trkTracking(folder_n, resultsFolder);
    
    % perform post-processing
    a = dir([resultsFolder '*' exp_num(i,:) '*.mat']);
    matFileName = a.name;
    disp(matFileName);
    if( exist([resultsFolder matFileName], 'file') > 0)
        R = load([resultsFolder matFileName]);
        R = trkPostProcessing(R);
        save([resultsFolder matFileName], '-struct', 'R');
    end
    
    toc
    matlabpool close
    disp('');
    disp('=============================================================')
    disp('');
end



% kill the matlab pool
%matlabpool close force



