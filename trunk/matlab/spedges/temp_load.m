path(path, ['../nucleus_detection/']);
train_settings;


tic; disp('...collecting and processing the TRAIN data set.');
TRAIN = vj_collect_data(train1, train0, 'size', IMSIZE, 'normalize', NORM, 'data_limit', [TRAIN_POS TRAIN_NEG]);toc;


% define a set of haar-like weak classifiers over the standard image size
tic; disp('...defining the haar-like weak classifiers.');
WEAK = vj_define_weak_classifiers(IMSIZE, 'types', [1 2 3 5]); toc; 
                    
%precompute the haar-like feature responses for each classifier over the
%training set and store them in a bigmatrix, PRE.
disp('...precomputing the haar-like feature responses of each classifier ');
disp(['   on the ' num2str(length(TRAIN)) ' training images (this may take quite some time).']);                        
PRE = vj_precompute_haar_response_new(TRAIN, WEAK, temp_filenm, temppath, []);



w = ones(1,length(TRAIN)) ./ length(TRAIN); 


t = 1; T = 1; training_labels = [TRAIN(:).class];
 %% 1. Normalize the weights
w = w ./sum(w);
wstring = ['       t=' num2str(t) '/' num2str(T) ' optimized feature '];
W = wristwatch('start', 'end', size(WEAK.descriptor,1), 'every', 10000, 'text', wstring);

%% 2. find the optimal class separating theta and minerr for each feature
for i = 1:(size(WEAK.descriptor,1))
    [WEAK, PRE] = vj_find_haar_parameters2(i, training_labels, PRE, WEAK, w);
    W = wristwatch(W, 'update', i);
end