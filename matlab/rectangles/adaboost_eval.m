if EVAL 
    % evaluate the strong classifier and record performance
    tic;
    if ANORM
        VALS = AdaBoostClassifyA_mex(CLASSIFIER.rects, CLASSIFIER.cols, CLASSIFIER.areas, CLASSIFIER.thresh, CLASSIFIER.pol, CLASSIFIER.alpha, D);    
    else
        VALS = AdaBoostClassify(CLASSIFIER.rects, CLASSIFIER.cols, CLASSIFIER.thresh, CLASSIFIER.pol, CLASSIFIER.alpha, D);
    end
    PR = single(VALS > 0); PR(PR == 0) = -1;
    [TP TN FP FN TPR FPR ACC] = rocstats(PR>0,L>0, 'TP', 'TN', 'FP', 'FN', 'TPR', 'FPR', 'ACC');
    stats(t,:) = [TP TN FP FN TPR FPR ACC]; to = toc;
    disp(['   TP = ' num2str(TP) '/' num2str(sum(L==1)) '  FP = ' num2str(FP) '/' num2str(sum(L==-1)) '  ACC = ' num2str(ACC)  '.  Elapsed time ' num2str(to) ' seconds.']);
end