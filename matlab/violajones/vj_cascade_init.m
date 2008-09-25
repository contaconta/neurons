function CASCADE = vj_cascade_init()
%
%
%
%
%
%
%

CASCADE.CLASSIFIER = [];            % the first classifier is left empty
CASCADE.fi = 1;                     % this classifiers false alarm rate
CASCADE.di = 1;                     % this classifiers detection rate
CASCADE.threshold = 2  ;            % the starting sliding adaboost threshold for this classifier
CASCADE.type = 'CASCADE';           % specify if this is a cascade or single classifier
