%load /osshare/Work/matlab_neuron_project/cvpr_tracking/GT.mat;

pos_train_path = '/osshare/Work/Data/nucleus_training_rotated/train/pos/';
neg_train_path = '/osshare/Work/Data/nucleus_training_rotated/train/neg/';
pos_test_path  = '/osshare/Work/Data/nucleus_training_rotated/test/pos/';
neg_test_path  = '/osshare/Work/Data/nucleus_training_rotated/test/neg/';

IMSIZE = [24 24];
ex_num = 1;
SIGMA = 2;


for t = 1:24
   
    % initialize the ground truth to be empty
    empty = zeros([1024 1024 3]);
    
    disp(['...computing edges for t = ' num2str(t)]);
    EDGES(:,:,t) = edge(GT(t).Image, 'log', 0, SIGMA);
    
    
    % loop through GT objects in the frame
    %for i = 1:length(GT(t).s)
    
    
end





