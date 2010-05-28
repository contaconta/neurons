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



ranks = zeros(1,T);

for t = 1:T
    ranks(t) = length(rects{t});
end

ranklist = unique(ranks);

rankhist = zeros(1, length(ranklist));
data = zeros(T, length(ranklist));



for t = 1:T
    
    thisrank = ranks(t);
    ind = find(ranklist == thisrank,1);
    rankhist(ind) = rankhist(ind) + 1;
    
    data(t,:) = (rankhist / sum(rankhist)) * 100;
    %data(t,:) = rankhist / sum(rankhist);
    
end

area(data);
colormap(summer);
grid on;
    
axis([1 T 0 100]);

legstr = {};
for i = 1:length(ranklist)
   legstr{i} = ['Rank ' num2str(ranklist(i))];
end
legend(legstr);

xlabel('Boosting Stage t');
ylabel('Classifier Rank (%)');

