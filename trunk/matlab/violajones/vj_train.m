% define paths to training images for each class
% train1 = '/osshare/Work/Data/boosting_training/train/neuron/';
% train0 = '/osshare/Work/Data/boosting_training/train/non-neuron/';
% test1 = '/osshare/Work/Data/boosting_training/test/neuron/';
% test0 = '/osshare/Work/Data/boosting_training/test/non-neuron/';

TRAIN_POS = 3000;     % number of positive examples in the training set
TRAIN_NEG = 5000;     % number of negative examples in the training set
TEST_POS = 3000;
TEST_NEG = 5000;

% path info
% ---------
matpath = [pwd '/mat/'];
train1 = '/osshare/Work/Data/face_databases/combined/train/face/';
train0 = '/osshare/Work/Data/face_databases/combined/train/non-face/';
test1 =  '/osshare/Work/Data/face_databases/combined/test/face/';
test0 =  '/osshare/Work/Data/face_databases/combined/test/non-face/';

IMSIZE = [19 19];  

% collect the training data into a struct
disp('...collecting and processing the training data.');
TRAIN = vj_collect_data(train1, train0, 'size', IMSIZE, 'save', ...
                        [matpath 'FACES_TRAIN.mat'], 'normalize', 1,   'data_limit', [TRAIN_POS TRAIN_NEG]);
           
% define a set of haar-like weak classifiers over the standard image size
disp('...defining the weak haar-like classifiers.');
WEAK = vj_define_weak_classifiers(IMSIZE, 'types', [1 2 3 5]);

% precompute the haar-like feature responses for each classifier over the
% training set and store them in a bigmatrix, PRE.
disp('...precomputing the haar-like feature responses of each classifier');
disp(['   on the ' num2str(length(TRAIN)) ' training images (this may take quite some time).']);
filenm = 'TEST_';       % file prefix for storing precomputed data                         
PRE = vj_precompute_haar_response(TRAIN, WEAK, filenm);


% now that the data is collected, features defined and precomuted, we will
% used ADABOOST to train a strong classifier from T weak classifier hypotheses
T = 10;
disp(['...training a T=' num2str(T) ' hypothesis boosted classifier.']);
CLASSIFIER = vj_adaboost(PRE, TRAIN,  WEAK, T);

% test the boosted classifier on the training data.  we
% will compute the true positive rate, false positive rate, and accuracy
gt = [TRAIN(:).class]';     % vector containing ground truth classes 
C = zeros(size(gt));        % vector containing our boosted classifier results
for i=1:length(TRAIN); C(i) = vj_classify_strong(CLASSIFIER, TRAIN(i).II); end
[tpr fpr acc] = rocstats(C, gt, 'TPR', 'FPR', 'ACC');  % evaluate results
disp(['Boosted classifier results on the training set: TPR=' num2str(tpr) ', FPR=' num2str(fpr) ', ACC=' num2str(acc) ]);


% now let's see what results we get on a test set.
TEST = vj_collect_data(test1, test0, 'size', IMSIZE, 'save', ...
                       [matpath 'FACES_TEST.mat'], 'normalize', 1, 'data_limit', [TEST_POS TEST_NEG]);

% test the boosted classifier on the test data. 
gt = [TEST(:).class]';     % vector containing ground truth classes 
C = zeros(size(gt));       % vector containing our boosted classifier results
for i=1:length(TEST); C(i) = vj_classify_strong(CLASSIFIER, TEST(i).II); end
[tpr fpr acc tnlist] = rocstats(C, gt, 'TPR', 'FPR', 'ACC', 'TNlist');  % evaluate results
disp(['Boosted classifier results on the test set: TPR=' num2str(tpr) ', FPR=' num2str(fpr) ', ACC=' num2str(acc) ]);

clear C gt i T matpath train1 test1 train0 test0 tnlist acc tpr fpr 
                   