function h = ada_spedge_classify(weak_learner, DATA, offset, IMSIZE)
%
%
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


%% perform the classification

if length(weak_learner) == 1
    % if we have a single weak learner, but many spedges
    f = ada_spedge_response(weak_learner.index, [DATA(:).sp]);
    if weak_learner.polarity == 1
        h = f < weak_learner.theta;
    else
        h = -f <= -weak_learner.theta;
    end
else
    % otherwise, we have many weak learners but a single spedge
    f = ada_spedge_response( {weak_learner.index}, DATA.sp);
    h = cellfun(@decision, num2cell(f), {weak_learner.polarity}, {weak_learner.theta});
end


%% function to make a classification decsision for a single weak learner
function h = decision(f, polarity, theta)

%h = polarity * f < polarity * theta;

if polarity == 1
    h = f < theta;
else
    h = f >= theta;
end
