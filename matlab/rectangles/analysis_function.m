function analysis_function(N, folder, IMSIZE)


fplocs = [1.1e-6:.1e-6:1e-5,1.1e-5:.1e-5:1e-4,1.1e-4:.1e-4:1e-3,1.1e-3:.1e-3:1e-2, 1.1e-2:.1e-2:1e-1, 1.1e-1:.1e-1:1];
count = 0;
load(['roc' num2str(N) '.mat']);    % load the existing copy



prefix = 'VJ-'; 
[TPvj,classifier_files,count] = analyze_group(TPvj, prefix, folder, count, N, fplocs, classifier_files);
save(['roc' num2str(N) '.mat']);    % save a temporary copy

prefix = 'A2-'; 
[TPa2,classifier_files,count] = analyze_group(TPa2, prefix, folder, count, N, fplocs, classifier_files);
save(['roc' num2str(N) '.mat']);    % save a temporary copy

prefix = 'A4-'; 
[TPa4,classifier_files,count] = analyze_group(TPa4, prefix, folder, count, N, fplocs, classifier_files);
save(['roc' num2str(N) '.mat']);    % save a temporary copy

prefix = 'A8-'; 
[TPa8,classifier_files,count] = analyze_group(TPa8, prefix, folder, count, N, fplocs, classifier_files);
save(['roc' num2str(N) '.mat']);    % save a temporary copy

prefix = 'A12-'; 
[TPa12,classifier_files,count] = analyze_group(TPa12, prefix, folder, count, N, fplocs, classifier_files);
save(['roc' num2str(N) '.mat']);    % save a temporary copy

prefix = '50-50'; 
[TP50,classifier_files,count] = analyze_group(TP50, prefix, folder, count, N, fplocs, classifier_files);
save(['roc' num2str(N) '.mat']);    % save a temporary copy

prefix = '33'; 
[TP33,classifier_files,count] = analyze_group(TP33, prefix, folder, count, N, fplocs, classifier_files);
save(['roc' num2str(N) '.mat']);    % save a temporary copy

prefix = 'VJANORM'; 
[TPvja,classifier_files,count] = analyze_group(TPvja, prefix, folder, count, N, fplocs, classifier_files);
save(['roc' num2str(N) '.mat']);    % save a temporary copy

prefix = 'VJDNORM';
[TPvjd,classifier_files,count] = analyze_group(TPvjd, prefix, folder, count, N, fplocs, classifier_files);
save(['roc' num2str(N) '.mat']);    % save a temporary copy

prefix = 'VJSPECIAL'; 
[TPvjs,classifier_files,count] = analyze_group(TPvjs, prefix, folder, count, N, fplocs, classifier_files);
save(['roc' num2str(N) '.mat']);    % save a temporary copy

prefix = 'Simple2-'; 
[TPs2,classifier_files,count] = analyze_group(TPs2, prefix, folder, count, N, fplocs, classifier_files);
save(['roc' num2str(N) '.mat']);    % save a temporary copy

prefix = 'Simple4-'; 
[TPs4,classifier_files,count] = analyze_group(TPs4, prefix, folder, count, N, fplocs, classifier_files);
save(['roc' num2str(N) '.mat']);    % save a temporary copy

prefix = 'Simple8-'; 
[TPs8,classifier_files,count] = analyze_group(TPs8, prefix, folder, count, N, fplocs, classifier_files);
save(['roc' num2str(N) '.mat']);    % save a temporary copy

prefix = 'Simple12-'; 
[TPs12,classifier_files,count] = analyze_group(TPs12, prefix, folder, count, N, fplocs, classifier_files);
save(['roc' num2str(N) '.mat']);    % save a temporary copy

prefix = 'Amix-'; 
[TPas10,classifier_files,count] = analyze_group(TPas10, prefix, folder, count, N, fplocs, classifier_files);
save(['roc' num2str(N) '.mat']);    % save a temporary copy

prefix = 'Amix25-'; 
[TPas25,classifier_files,count] = analyze_group(TPas25, prefix, folder, count, N, fplocs, classifier_files);
save(['roc' num2str(N) '.mat']);    % save a temporary copy

prefix = 'Amix33-'; 
[TPas33,classifier_files,count] = analyze_group(TPas33, prefix, folder, count, N, fplocs, classifier_files);
save(['roc' num2str(N) '.mat']);    % save a temporary copy

prefix = 'Amix50-'; 
[TPas50,classifier_files,count] = analyze_group(TPas50, prefix, folder, count, N, fplocs, classifier_files);
save(['roc' num2str(N) '.mat']);    % save a temporary copy

prefix = 'Amix25Disconnect-'; 
[TPas25d,classifier_files,count] = analyze_group(TPas25d, prefix, folder, count, N, fplocs, classifier_files);
save(['roc' num2str(N) '.mat']);    % save a temporary copy

plot(fplocs, mean(TPvj,1), 'k', 'LineWidth', 2);
plot(fplocs, mean(TPa2,1), 'b');
plot(fplocs, mean(TPa4,1), 'r');
plot(fplocs, mean(TPa8,1), 'g');
plot(fplocs, mean(TPa12,1), 'y');
plot(fplocs, mean(TP50,1), 'c', 'LineWidth', 2);
plot(fplocs, mean(TP33,1), 'm', 'LineWidth', 2);
plot(fplocs, mean(TPvja,1), 'b--', 'LineWidth', 2);
plot(fplocs, mean(TPvjd,1), 'r--', 'LineWidth', 2);
plot(fplocs, mean(TPvjspecial,1), 'g--', 'LineWidth', 2);
plot(fplocs, mean(TPs2,1), 'b:');
plot(fplocs, mean(TPs4,1), 'r:');
plot(fplocs, mean(TPs8,1), 'g:');
plot(fplocs, mean(TPs12,1), 'y-.');
plot(fplocs, mean(TPas10,1), 'r-.');
plot(fplocs, mean(TPas25,1), 'g-.');
plot(fplocs, mean(TPas33,1), 'b-.');
plot(fplocs, mean(TPas50,1), 'c-.');
plot(fplocs, mean(TPas25d,1), 'm-.');

legend('VJ', 'A2', 'A4', 'A8', 'A12', '50-50', '33', 'ANORM', 'ADNORM', 'VJ-SPECIAL', 'S2', 'S4', 'S8', 'S12', 'ASYMM10', 'ASYMM25', 'ASYMM33', 'ASYMM50', 'ASYMM25D');
title(['Area Normalized vs Viola-Jones ' num2str(NLEARNERS) ' learners, ' num2str(NP) ' (+) / ' num2str(NN) ' (-) examples.']);


keyboard;





function  [TPout, classifier_files,count] = analyze_group(TPout, prefix, folder, count, NLEARNERS, fplocs, classifier_files)


d = dir([folder prefix '*']);
for i = 1:length(d)
    filename = d(i).name;
    if ~ismember(filename, classifier_files);   % if it is not a member we should process it
        load([folder filename]);
        if length(CLASSIFIER.rects) >= NLEARNERS    % we should only process if it has sufficient learners
            [TP FP NP NN] = evaluate_test_set(CLASSIFIER, NLEARNERS, filename);
            TPout(count,:) = interp1(FP/NN,TP/NP,fplocs); %#ok<AGROW>
            count = count + 1; classifier_files{count} = filename;  %#ok<AGROW>
        end
    end
end
