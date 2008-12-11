function h = ada_haar_classifytr(inds, weak_learner, DATA)
%
%  h = ada_haar_classify(weak_learner, DATA)
%
%
%
%
%


polarity = [weak_learner(:).polarity];
theta = [weak_learner(:).theta];

f = DATA.responses.getRows(inds);


h =  (polarity.*f < polarity.*theta)';


% %% perform the classification
% 
% if length(weak_learner) == 1
%     % if we have a single weak learner, but many integral images
%     f = ada_haar_response(weak_learner.hinds, weak_learner.hvals, II);
%     h = (weak_learner.polarity*ones(size(f)) .* f) <  ((weak_learner.polarity*ones(size(f))) .* (weak_learner.theta*ones(size(f))));
% else
%     % otherwise, we have many weak learners but a single integral image 
%     f = ada_haar_response( {weak_learner.hinds}, {weak_learner.hvals}, II);
%     h = cellfun(@decision, num2cell(f), {weak_learner.polarity}, {weak_learner.theta});
% end
% 
% 
% 
% 
% %% function to make a classification decsision for a single weak learner
% function h = decision(f, polarity, theta)
% 
% h = polarity * f < polarity * theta;
