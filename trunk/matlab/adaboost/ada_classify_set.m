function C = ada_classify_set(CLASSIFIER, SET, varargin)
%
%
%   ada_classify_set(CASCADE, TRAIN);
%   ada_classify_set(CLASSIFIER, VALIDATION, threshold)
%
%

    

% if we are passed a single stage instead of a cascade, make it appear as a
% similar structure to a cascade
if strcmp(CLASSIFIER.type, 'SINGLE');
    CASCADE.CLASSIFIER = CLASSIFIER;
else
    CASCADE = CLASSIFIER;
end


% initialize the classification vector to be all positive classes
C =  ones(size(SET.class));


% loop through each stage of the cascade (even if it is only 1 stage)
for s = 1:length(CASCADE)
   
    % load the data needed for the current stage of the cascade.  each
    % column contains the responses of one feature_index
    f = SET.responses.getCols(CASCADE(s).CLASSIFIER.feature_index);
    
    % make a list of examples which are still positive
    positive_list = find(boolean(C));
    
    % handle the case where there are no positive examples left
    if isempty(positive_list); C = zeros(size(C)); disp('no positives left!'); break; end
    
    % we only need feature responses for the positive examples
    f = f(positive_list, :);
    
    % each row of h contains weak classifier responses to example p
%     for p = 1:length(positive_list)
%         h(p,:) = weak_classify(CASCADE(s).CLASSIFIER,f(p,:));
%     end
    
    h = weak_classify(CASCADE(s).CLASSIFIER,f);
    
    
    %% compute the strong classification
    % the strong classification is computed by the weighted sum of the weak
    % classification results.  if this is > .5 * sum(alpha), it is class 1,
    % otherwise it is class 0.  varying 'threshold' (default = 1) adjusts 
    % the sensitivity of the strong classifier
    alpha = repmat(CASCADE(s).CLASSIFIER.alpha, length(positive_list), 1);
    ha = h .* alpha;
    asum = sum(alpha,2);
    
    if nargin > 2; threshold = varargin{1}; else threshold = CASCADE(s).threshold; end
    
    % Cstage is a column vector containing strong classification results
    % for stage s.  each row represents the classification of a training
    % example
    Cstage = zeros(size(C));
    Cstage(positive_list) = sum(ha,2) > (.5 * asum * threshold);

    % an AND operation updates newly found rejections
    C = C & Cstage;
    
end


%% each contains a row vector of weak classifications given responses in f
function h = weak_classify(CLASSIFIER, f)


polarity = zeros(1,size(f,2));
theta = zeros(1,size(f,2));

learner_types = unique(CLASSIFIER.learner_type);
for l = 1:length(learner_types)
    type = CLASSIFIER.learner_type{l};
    pol = [CLASSIFIER.(type).polarity];
    the = [CLASSIFIER.(type).theta];
    
    inds = find(strcmp(CLASSIFIER.learner_type, type));
    polarity(inds) = pol;
    theta(inds) = the;
end

polarity = repmat(polarity, size(f,1), 1);
theta = repmat(theta, size(f,1),1);

h = polarity .* f < polarity .* theta;



% %% each contains a row vector of weak classifications given responses in f
% function h = weak_classify(CLASSIFIER, f)
% 
% 
% polarity = zeros(size(f));
% theta = zeros(size(f));
% 
% learner_types = unique(CLASSIFIER.learner_type);
% for l = 1:length(learner_types)
%     type = CLASSIFIER.learner_type{l};
%     pol = [CLASSIFIER.(type).polarity];
%     the = [CLASSIFIER.(type).theta];
%     
%     inds = find(strcmp(CLASSIFIER.learner_type, type));
%     polarity(inds) = pol;
%     theta(inds) = the;
% end
