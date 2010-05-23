function FEATURES = haar_feature(D, INDS, POLS)


FEATURES = zeros(size(D,1),1);

% for each rectangle that composes the feature
for r = 1:length(POLS)
    
    ind = INDS{r};  % the rectangle corner indexes [1 2 3 4]
    pol = POLS(r);  % the polarity of the rectangle
    
    
    FEATURES(:,1) = FEATURES(:,1) + pol * (D(:,ind(1)) + D(:,ind(4)) - D(:,ind(3)) - D(:,ind(2)));
    
    
end
