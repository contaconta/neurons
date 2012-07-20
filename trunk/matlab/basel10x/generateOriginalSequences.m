function generateOriginalSequences(folder, resultsFolder)

% --------- generate list of folders to process -----------
count = 1;
listOfDirs = dir(folder);
for i = 1:length(listOfDirs)
    if listOfDirs(i).isdir && length(listOfDirs(i).name) > 2
        exp_num(count,:) = listOfDirs(i).name; %#ok<*AGROW>
        count  = count + 1;
    end
end


% ------------ process the specified folders --------------
for i = 1:size(exp_num,1)
    
    tic
    folder_n = [folder exp_num(i,:) '/'];
    trkGenerateSequencesforAnnotations(folder_n, resultsFolder, exp_num(i,:));
    toc
    disp('');
    disp('=============================================================')
    disp('');
end
