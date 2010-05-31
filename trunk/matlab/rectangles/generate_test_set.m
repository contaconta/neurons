%function [TESTD, TESTL] = generate_test_set(pos_folder, neg_folder, NPOS, N, IMSIZE)

IMSIZE = [24 24];

DATA_FOLDER = '/osshare/Work/Data/face_databases/EPFL-CVLAB_faceDB/';
%pos_train_folder = [DATA_FOLDER 'train/pos/'];
%neg_train_folder = [DATA_FOLDER 'non-face_uncropped_images/'];
pos_folder = [DATA_FOLDER 'test/pos/'];
neg_folder = [DATA_FOLDER 'non-face_uncropped_images/'];


%% the positive set
NPOS = 5000;

[Li,Di] = collect_positive_examples(NPOS, IMSIZE, pos_folder); 
NPOS = length(Li);

filename = ['TEST1' '.mat'];
disp(['...storing ' num2str(sum(Li==1)) ' (class +1) / ' num2str(sum(Li==-1)) ' (class -1) examples to ' filename]);
save(filename, '-v7.3', 'Li', 'Di');  

%% the negative set
NNEGi = 100000;
Ni = 10;

for i = 1:Ni

    [Li,Di] = collect_negative_examples(NNEGi, IMSIZE, neg_folder);

    filename = ['TEST' num2str(i+1) '.mat'];
    disp(['...storing ' num2str(sum(Li==1)) ' (class +1) / ' num2str(sum(Li==-1)) ' (class -1) examples to ' filename]);
    save(filename, '-v7.3', 'Li', 'Di');  

end


% TESTD = [Dp;Dn];  clear Dp Dn;  % D contains all integral image data (each row contains a vectorized image)
% TESTL = [Lp;Ln];  clear Lp Ln;  % L contains all associated labels