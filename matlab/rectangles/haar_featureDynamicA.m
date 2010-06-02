function FEATURES = haar_featureDynamicA(D, rects, cols, areas)
%   given a single haar learner, rectS
%
%
%



FEATURES_1 = zeros(size(D,1),1);
FEATURES_0 = zeros(size(D,1),1);

% for each rectangle that composes the feature
for r = 1:length(cols)
    
    rect = rects{r};  % the rectangle corner rectexes [1 2 3 4]
    col = cols(r);  % the polarity of the rectangle
    
    if col == 1
        FEATURES_1(:,1) = FEATURES_1(:,1) + (D(:,rect(1)) + D(:,rect(4)) - D(:,rect(3)) - D(:,rect(2)));
    else
        FEATURES_0(:,1) = FEATURES_0(:,1) + (D(:,rect(1)) + D(:,rect(4)) - D(:,rect(3)) - D(:,rect(2)));
    end    
end


%AREA_1 = sum(areas(cols == 1));
%AREA_0 = sum(areas(cols == -1));

AREA_1 = areas(1);
AREA_0 = areas(2);

if AREA_1 ~= 0
    FEATURES_1 = FEATURES_1 / AREA_1;
end

if AREA_0 ~= 0
    FEATURES_0 = FEATURES_0 / AREA_0;
end

FEATURES = FEATURES_1 - FEATURES_0;