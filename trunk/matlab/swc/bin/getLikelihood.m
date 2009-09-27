function [predicted_label, accuracy, pb, fv] = getLikelihood(Y, model, minI,maxI, rescaleData)
% Y is a matrix of pixels whose values has to within the range {0,255}.
% Output : probability of the input set of pixels to belong to a mitochondria

nbBins = 20.0;

% Build feature vector
bins = zeros(1,nbBins);
for i=1:length(Y)
  idx = floor((double(Y(i))*nbBins)/255.0)+1; %FIXME : 256
  if idx > 20
    disp('Use 8 bit-images only')
    keyboard
  end
  bins(idx) = bins(idx) + 1;
end

testing_label = 0; % dummy values

% Output feature vector
% Check if data need to be rescaled
if rescaleData == false
  fv = bins;
else
  fv = (bins - repmat(minI,size(bins,1),1))*spdiags(1./(maxI-minI)',0,size(bins,2),size(bins,2));
  fv(find(isnan(fv)))=0;
  fv(find(isinf(fv)))=0;
end

% Prediction
[predicted_label, accuracy, pb] = svmpredict(testing_label, fv, model, '-b 1 ');
