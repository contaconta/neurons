function h = ada_haar_classify(weak_learner, DATA, offset, IMSIZE)
%
%  h = ada_haar_classify(weak_learner, DATA, offset, IMSIZE)
%
%
%
%
%

%% handle input parameters
x = offset(1);
y = offset(2);

if nargin < 3
    x = 0;  y = 0;
end


%% extract the integral image(s) from DATA
if length(DATA) > 1
    II = [DATA(:).II];
else
    II = DATA.II;
    % vectorize the intergral image if it is not already
    if size(II,2) ~= 1
        II = II(y+1:y+IMSIZE(2), x+1:x+IMSIZE(1));
        II = II(:);
    end
end



%% perform the classification

if length(weak_learner) == 1
    % if we have a single weak learner, but many integral images
    f = ada_haar_response(weak_learner.hinds, weak_learner.hvals, II);
    h = (weak_learner.polarity*ones(size(f)) .* f) <  ((weak_learner.polarity*ones(size(f))) .* (weak_learner.theta*ones(size(f))));
else
    % otherwise, we have many weak learners but a single integral image 
    f = ada_haar_response( {weak_learner.hinds}, {weak_learner.hvals}, II);
    h = cellfun(@decision, num2cell(f), {weak_learner.polarity}, {weak_learner.theta});
end





%% function to make a classification decsision for a single weak learner
function h = decision(f, polarity, theta)

h = polarity * f < polarity * theta;


