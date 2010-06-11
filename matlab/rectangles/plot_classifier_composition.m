function plot_classifier_composition(CLASSIFIER)
%
%
%
%

T = length(CLASSIFIER.rects);

rects = CLASSIFIER.rects(1:T);
thresh = CLASSIFIER.thresh(1:T);
alpha = CLASSIFIER.alpha(1:T);

% backwards-compatible naming
if isfield(CLASSIFIER, 'pols')
    cols = CLASSIFIER.pols(1:T);
else
    cols  = CLASSIFIER.cols(1:T);
end
if isfield(CLASSIFIER, 'tpol')
    pol = CLASSIFIER.tpol(1:T);
else
    pol = CLASSIFIER.pol(1:T);
end
if isfield(CLASSIFIER, 'areas');
    ANORM = 1;
    areas = CLASSIFIER.areas(1:T);
else
    ANORM = 0;
end
if isfield(CLASSIFIER, 'types');
    methods = CLASSIFIER.types(1:T);
else
    methodname = {input('no methods specified, please type a name for the feature generation method:\n', 's')};
    methods(1:T) = deal(methodname);
end


methodlist = unique(methods);

mcount = 0;

for m = 1:length(methodlist)
    
    ranks = zeros(1,T);

    for t = 1:T
        ranks(t) = length(rects{t});
    end

    ranklist = unique(ranks);


    for r = 1:length(ranklist)
        mcount = mcount + 1;
        labels{mcount} = [methodlist{m} ' Rank ' num2str(ranklist(r))];
        ID{mcount} = [num2str(m) ' ' num2str(r)];
    end
    
    
    
    %rankhist = zeros(1, length(ranklist));
    %data = zeros(T, length(ranklist));

end

methodhist = zeros(1,mcount);
data = zeros(t,mcount);

%keyboard;


for t = 1:T

    thisrank = ranks(t);
    thismethod = methods{t};

    % determine the ID
    [x methodind] =ismember(thismethod, methodlist);
    rankind = find(ranklist == thisrank,1);
    
    mind = find( ismember(ID, [num2str(methodind) ' ' num2str(rankind)]));
    methodhist(mind) = methodhist(mind) + 1;
    
    %rankhist(ind) = rankhist(ind) + 1;
    data(t,:) = (methodhist / sum(methodhist)) * 100;
    
    %data(t,:) = (rankhist / sum(rankhist)) * 100;
 

end

nonzeroinds = find(methodhist);
data = data(:,nonzeroinds);
labels = labels(nonzeroinds);

    
area(data);
colormap(summer);
grid on;
    
axis([1 T 0 100]);

% legstr = {};
% for i = 1:length(ranklist)
%    legstr{i} = ['Rank ' num2str(ranklist(i))];
% end
% legend(legstr);
legend(labels);

xlabel('Boosting Stage (t)');
ylabel('Percent of Weak Learners (%)');


BACKGROUNDCOLOR = [1 1 1];
set(gcf, 'Color', BACKGROUNDCOLOR);

%keyboard;