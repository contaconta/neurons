function [model, minI, maxI] = svm_model()


% set necessary paths
%addpath('../utils/libsvm-mat-2.89-3');
%addpath([pwd '/../utils/']);
%huttPath = '/osshare/Work/software/huttenlocher_segment/';
%imPath = [pwd '/../images/'];
imPath = '/osshare/Work/Data/LabelMe/Images/fibsem/';
imName = 'FIBSLICE0720';
feature_vectors = [pwd '/temp/Model-0-4200-3-sup/feature_vectors'];

[label_vector, instance_matrix] = libsvmread(feature_vectors);
training_label = label_vector(1:4000,:);
training_instance = instance_matrix(1:4000,:);
%testing_label = label_vector(3001:size(label_vector,1),:);
%testing_instance = instance_matrix(3001:size(instance_matrix,1),:);

% if ~exist('model')
%   disp('Computing model...');
  [model,minI,maxI] = loadModel(training_label, training_instance);
% end