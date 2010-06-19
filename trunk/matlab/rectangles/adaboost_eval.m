

if EVAL 

    % evaluate the strong classifier and record performance
    tic;
    
    if OPT_WEIGHTS
        E = AdaBoostClassifyOPT_WEIGHTS_mex(CLASSIFIER.rects, CLASSIFIER.weights, CLASSIFIER.separeas, CLASSIFIER.thresh, CLASSIFIER.pol, CLASSIFIER.alpha, D);
    else        
        E = AdaBoostClassifyDynamicA_mex(CLASSIFIER.rects, CLASSIFIER.cols, CLASSIFIER.areas, CLASSIFIER.thresh, CLASSIFIER.pol, CLASSIFIER.alpha, D);
    end
    
    
    %E = AdaBoostClassifyDynamicA_mex(CLASSIFIER.rects, CLASSIFIER.cols, CLASSIFIER.areas, CLASSIFIER.thresh, CLASSIFIER.pol, CLASSIFIER.alpha, D);
    
    E = single(E > 0); E(E == 0) = -1;

    [TP TN FP FN TPR FPR ACC] = rocstats(E>0,L>0, 'TP', 'TN', 'FP', 'FN', 'TPR', 'FPR', 'ACC');
    stats(t,:) = [TP TN FP FN TPR FPR ACC]; to = toc;
    disp(['   TP = ' num2str(TP) '/' num2str(sum(L==1)) '  FP = ' num2str(FP) '/' num2str(sum(L==-1)) '  ACC = ' num2str(ACC)  '.  Elapsed time ' num2str(to) ' seconds.']);
end