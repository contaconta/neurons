function adaboost_self_evaluate(mat_filename)

fplocs = [1.1e-6:.1e-6:1e-5,1.1e-5:.1e-5:1e-4,1.1e-4:.1e-4:1e-3,1.1e-3:.1e-3:1e-2, 1.1e-2:.1e-2:1e-1, 1.1e-1:.1e-1:1];
Tlist = [50 100 200 400 600 800 1000 1200 1400 1600 1800 2000];
%Tlist = [50 100 200];

[pathstr,name,ext,versn] = fileparts(mat_filename);


disp(' ');
disp(['--------- evaluating ' name '---------']);

% load the mat file containing the classifier
load([pathstr '/' name ext]);


NTestSets = 11;
L = [ ]; VALS = cell(1, length(Tlist));

for i = 1:NTestSets
    
    % load part i of the data set
    if strcmp(CLASSIFIER.method, 'Lienhart'); di_filename = ['45_TEST' num2str(i) '.mat'];
    else; di_filename = ['TEST' num2str(i) '.mat']; end;    disp(['...loading ' di_filename]); %#ok<NOSEM>
    load(di_filename);
    
    % collect the labels
    L = [L; Li]; %#ok<AGROW>
    
    fprintf('   classifying for T = ');
    for t = 1:length(Tlist)
        T = Tlist(t);
        rects = CLASSIFIER.rects(1:T);
        thresh = CLASSIFIER.thresh(1:T);
        alpha = CLASSIFIER.alpha(1:T);
        cols  = CLASSIFIER.cols(1:T);
        pol = CLASSIFIER.pol(1:T);
        areas = CLASSIFIER.areas(1:T);
        fprintf('%s ', num2str(T));
        VALSi = AdaBoostClassifyDynamicA_mex(rects, cols, areas, thresh, pol, alpha, Di);
        
        VALS{t} = [VALS{t}; VALSi];
    end
    fprintf('\n');
    
    
end



TPR_list = zeros(length(Tlist), length(fplocs));

NP = sum(L == 1);
NN = sum(L == -1);
fprintf('...evaluating the ROC for T = ');
for t = 1:length(Tlist)
    fprintf('%s ', num2str(Tlist(t)));
    [TP FP] = roc_evaluate2(VALS{t}, L);
    TPR_list(t,:) = interp1(FP/NN,TP/NP,fplocs);
end
fprintf('\n');

save([pathstr '/' name ext], 'CLASSIFIER', 'W', 'error', 'TPR_list', 'fplocs', 'Tlist');
disp(['... saved results in ' pathstr '/' name ext ]);

