function [TP FP NP NN] = evaluate_test_set(CLASSIFIER, T, filename)

disp(' ');
disp(['--------- evaluating ' filename '---------']);

NTestSets = 11;  IMSIZE = [24 24];

rects = CLASSIFIER.rects(1:T);
thresh = CLASSIFIER.thresh(1:T);
alpha = CLASSIFIER.alpha(1:T);

% backwards-compatible naming
if isfield(CLASSIFIER, 'pols')
    cols = CLASSIFIER.pols(1:T);
else
    cols  = CLASSIFIER.cols(1:T);
end
if isfield(CLASSIFIER, 'tpol')
    pol = CLASSIFIER.tpol(1:T);
else
    pol = CLASSIFIER.pol(1:T);
end
% if isfield(CLASSIFIER, 'areas');
%     ANORM = 1;
%     areas = CLASSIFIER.areas(1:T);
% else
%     ANORM = 0;
% end
if isfield(CLASSIFIER, 'areas');
    areas = CLASSIFIER.areas(1:T);
else
    areas = compute_areas2(IMSIZE, rects, cols);
end

L = [ ]; VALS = [ ];

for i = 1:NTestSets
    
    filename = ['TEST' num2str(i) '.mat'];
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