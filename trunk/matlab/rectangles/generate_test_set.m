function [TESTD, TESTL] = generate_test_set(pos_folder, neg_folder, NPOS, N, IMSIZE)


[Lp,Dp] = collect_positive_examples(NPOS, IMSIZE, pos_folder); 
NPOS = length(Lp);

[Ln,Dn] = collect_negative_examples(N-NPOS, IMSIZE, neg_folder);

TESTD = [Dp;Dn];  clear Dp Dn;  % D contains all integral image data (each row contains a vectorized image)
TESTL = [Lp;Ln];  clear Lp Ln;  % L contains all associated labels

save('TEST.mat', '-v7.3', 'TESTD', 'TESTL');  
disp(['...storing ' num2str(sum(TESTL==1)) ' (class +1) / ' num2str(sum(TESTL==-1)) ' (class -1) examples to TEST.mat.']);