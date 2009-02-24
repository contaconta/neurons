%% load our settings, make sure they reflect the experiment!
ada_settings; 
ada_versioninfo;

%%% create the test set
% FILES.test_filenm    = [pwd '/mat/TESTD_FEATURES.dat'];
% DATASETS.VALIDATION_NEG = 5000;
% TEST = ada_collect_data(DATASETS, 'populate');

%% define the features this experiment used
WEAK = ada_define_weak_learners(LEARNERS); toc; 

%% precompute responses to the test set
TEST = ada_precompute(TEST, LEARNERS, WEAK, FILES, FILES.test_filenm);

%% load the saved cascaded classifier
load HA-nuclei24-23-Feb-2009-20.41.11.mat;
%load SP-sobel-nuclei24-23-Feb-2009-20.41.22.mat

%% compute the classifiers response to the test set
FILES.log_filenm = 'HAAR.log';

logfile(FILES.log_filenm, 'erase');logfile(FILES.log_filenm, 'header', {INFO.appname, ['Version ' INFO.version], ['by ' INFO.author ', ' INFO.email], [num2str(TRAIN_POS) ' positive examples, ' num2str(TRAIN_NEG) ' negative examples.'], ['DATASETS from ' DATASETS.filelist], ['LEARNERS ' LEARNERS(:).feature_type],['Started at ' datestr(now)], INFO.copyright, '-----------------------------------'});
logfile(FILES.log_filenm, 'column_labels', {'stage', 'step', 'Weak ID', 'Di', 'Fi', 'di', 'fi', 'di(train)', 'fi(train)', 'LEARNER'});


learners = 1;
for i = 1:length(CASCADE);
   
    for ti = 1:length(CASCADE(i).CLASSIFIER.feature_index)
    
        CAS_TEMP = CASCADE(1:i);
        CAS_TEMP(i).CLASSIFIER.feature_index = CAS_TEMP(i).CLASSIFIER.feature_index(1:ti);
        CAS_TEMP(i).CLASSIFIER.polarity = CAS_TEMP(i).CLASSIFIER.polarity(1:ti);
        CAS_TEMP(i).CLASSIFIER.theta = CAS_TEMP(i).CLASSIFIER.theta(1:ti);
        CAS_TEMP(i).CLASSIFIER.alpha = CAS_TEMP(i).CLASSIFIER.alpha(1:ti);
        CAS_TEMP(i).CLASSIFIER.weak_learners = CAS_TEMP(i).CLASSIFIER.weak_learners{1:ti};
        
        gt = [TEST(:).class]'; 
        C = ada_classify_set(CAS_TEMP, TEST);
        [tpr fpr FPs] = rocstats(C, gt, 'TPR', 'FPR', 'FPlist');
        disp([num2str(learners) ' learners, Di=' num2str(tpr) ', Fi=' num2str(fpr) ', #FP = ' num2str(length(FPs)) '.  CASCADE stage ' num2str(i) ' learner ' num2str(ti) ' applied to TEST set.'  ]);               
        
        
        for l = 1:length(LEARNERS); if strcmp(CASCADE(i).CLASSIFIER.weak_learners{ti}.type, LEARNERS(l).feature_type); L_ind = l; end; end;
        %logfile(FILES.log_filenm, 'write', [i ti CASCADE(i).CLASSIFIER.feature_index(ti) tpr fpr CASCADE(i).di CASCADE(i).fi tpr fpr  L_ind]);
        logfile(FILES.log_filenm, 'write', [i ti CASCADE(i).CLASSIFIER.feature_index(ti) tpr fpr 0 0 0 0  L_ind]);
        
        if learners > 200;
            return
        end
        
        learners = learners + 1;
    end
    
end