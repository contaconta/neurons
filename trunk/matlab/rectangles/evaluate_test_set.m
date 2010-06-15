function [TP FP NP NN] = evaluate_test_set(CLASSIFIER, T, filename)

disp(' ');
disp(['--------- evaluating ' filename '---------']);

NTestSets = 11;  

rects = CLASSIFIER.rects(1:T);
thresh = CLASSIFIER.thresh(1:T);
alpha = CLASSIFIER.alpha(1:T);
cols  = CLASSIFIER.cols(1:T);
pol = CLASSIFIER.pol(1:T);
areas = CLASSIFIER.areas(1:T);


L = [ ]; VALS = [ ];

for i = 1:NTestSets

    if strcmp(CLASSIFIER.method, 'Lienhart')
        filename = ['45_TEST' num2str(i) '.mat'];
    else
        filename = ['TEST' num2str(i) '.mat'];
    end
    disp(['...loading ' filename]);
    load(filename);
    
    L = [L; Li]; %#ok<AGROW>
    
    tic; disp(['   collecting classifier responses to ' filename]); pause(0.002);
    
    VALSi = AdaBoostClassifyDynamicA_mex(rects, cols, areas, thresh, pol, alpha, Di);

    to=toc;  disp(['   Elapsed time ' num2str(to) ' seconds.']);

    VALS = [VALS; VALSi]; %#ok<AGROW>

end

[TP FP] = roc_evaluate2(VALS, L);

NP = sum(L == 1);
NN = sum(L == -1);