function [model,minI,maxI] = loadModel(label, instance, rescaleData, kernelType)
% - label is a vector containing the labels associated to the instance matrix
% - instance is a matrix composed of the feature vectors for all the samples used for training
% - rescaleData is a boolean whose value specifies if feature vectors have to be rescaled
% - kernelType : set type of kernel function (default 2)
%	0 -- linear: u'*v
%	1 -- polynomial: (gamma*u'*v + coef0)^degree
%	2 -- radial basis function: exp(-gamma*|u-v|^2)
%	3 -- sigmoid: tanh(gamma*u'*v + coef0)
%	4 -- precomputed kernel (kernel values in training_set_file)
% Outputs :
% - model storing the support vectors used for prediction
% - minI : vector containing the min value for each element of the feature vectors
% - maxI : vector containing the min value for each element of the feature vectors

% scale each feature to the range of [0.1]
if rescaleData
  T2 = zeros(size(instance));
  m = max(instance);
  for i=1:size(instance,2)
    if m(i) ~= 0
      T2(:,i) = instance(:,i)/m(i);
    end
  end

  minI = min(instance,[],1);
  maxI = max(instance,[],1);
  T = (instance - repmat(minI,size(instance,1),1))*spdiags(1./(maxI-minI)',0,size(instance,2),size(instance,2));

  T(isnan(T))=0;
  T(isinf(T))=0;
  
  %keyboard
  
else
  T = instance;
  minI = 0;
  maxI = 0;
end

% Parameter selection
doParamSelection = true;
if doParamSelection
  bestcv = 0;
  for log2c = -1:3,
    for log2g = -4:1,
      cmd = ['-v 5 -t ' num2str(kernelType) ' -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)];
      cv = svmtrain(label, T, cmd);
      if (cv >= bestcv),
        bestcv = cv; bestc = 2^log2c; bestg = 2^log2g;
      end
      fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', log2c, log2g, cv, bestc, bestg, bestcv);
    end
  end
  model = svmtrain(label, T, ['-b 1 -t ' num2str(kernelType) ' -c ' num2str(bestc) ' -g ' num2str(bestg)]);
else
  model = svmtrain(label, T, ['-b 1 -t ' num2str(kernelType) ' -c 8 -g 2']); %#ok<UNRCH>
end
