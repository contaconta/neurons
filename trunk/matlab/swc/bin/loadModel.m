function [model,minI,maxI] = loadModel(label, instance)
% Y is a matrix of pixels whose values has to within the range {0,255}.
% Output : probability of the input set of pixels to belong to a mitochondria

rescaleMat = true;

% scale each feature to the range of [0.1]
if rescaleMat
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

  T(find(isnan(T)))=0;
  T(find(isinf(T)))=0;
  
  %keyboard
  
else
  T = instance;
end

%model = svmtrain(label, T, '-b 1 -c 8 -g 0.5');

% Parameter selection
doParamSelection = false;
if doParamSelection
  bestcv = 0;
  for log2c = -1:3,
    for log2g = -4:1,
      cmd = ['-v 5 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)];
      cv = svmtrain(label, T, cmd);
      if (cv >= bestcv),
        bestcv = cv; bestc = 2^log2c; bestg = 2^log2g;
      end
      fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', log2c, log2g, cv, bestc, bestg, bestcv);
    end
  end
  model = svmtrain(label, T, ['-b 1 -c ' num2str(bestc) ' -g ' num2str(bestg)]);
else
  model = svmtrain(label, T, '-b 1 -c 8 -g 2');
end