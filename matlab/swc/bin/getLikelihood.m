function [predicted_label, accuracy, pb] = getLikelihood(Y, model, minI,maxI)
% Y is a matrix of pixels whose values has to within the range {0,255}.
% Output : probability of the input set of pixels to belong to a mitochondria

addpath('../utils/libsvm-mat-2.89-3');
nbBins = 20.0;

% Build feature vector
%n = hist(double(Y),nbBins);

n = zeros(1,nbBins);
for i=1:length(Y)
  idx = floor((double(Y(i))*nbBins)/255.0)+1;
  if idx > 20
    disp('Use 8 bit-images only')
    keyboard
  end
  n(idx) = n(idx) + 1;
end

testing_label = 0; % dummy values

% normalize so that max value becomes 1
n
sum(n)
%testing_instance_matrix = (n/max(n))*2 - 1
%testing_instance_matrix = n./m
%testing_instance_matrix = n;
T = (n - repmat(minI,size(n,1),1))*spdiags(1./(maxI-minI)',0,size(n,2),size(n,2));

T(find(isnan(T)))=0;
T(find(isinf(T)))=0

% Prediction
[predicted_label, accuracy, pb] = svmpredict(testing_label, T, model, '-b 1');
