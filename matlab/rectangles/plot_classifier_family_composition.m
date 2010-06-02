function plot_classifier_family_composition(CLASSIFIER)
%
%
%
%

T = length(CLASSIFIER.rects);

rects = CLASSIFIER.rects(1:T);
thresh = CLASSIFIER.thresh(1:T);
alpha = CLASSIFIER.alpha(1:T);
family = CLASSIFIER.types(1:T);

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

for t = 1:T
    if isempty(family{t})
        family{t} = 'Karim1';
    end
end


famlist = unique(family);

famhist = zeros(1, length(famlist));
data = zeros(T, length(famlist));



for t = 1:T
    
    thisfam = family(t);
    [V,ind] = ismember(thisfam, famlist);
    famhist(ind) = famhist(ind) + 1;
    
    data(t,:) = (famhist / sum(famhist)) * 100;
    %data(t,:) = rankhist / sum(rankhist);
    
end

area(data);
colormap(summer);
grid on;
    
axis([1 T 0 100]);

legstr = {};
for i = 1:length(famlist)
   legstr{i} = ['Family ' famlist{i}];
end
legend(legstr);

xlabel('Boosting Stage t');
ylabel('Classifier Family (%)');
