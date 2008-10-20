


CTEST = ada_cascade_init;
CTEST.CLASSIFIER.feature_index = CASCADE(1).CLASSIFIER.feature_index;
CTEST.CLASSIFIER.alpha = CASCADE(1).CLASSIFIER.alpha;
CTEST.CLASSIFIER.haar(1).descriptor = [];
CTEST.CLASSIFIER.haar(1).hinds = [];
CTEST.CLASSIFIER.haar(1).hvals = [];

for i = 1:length(CTEST.CLASSIFIER.feature_index)
    f_ind = CTEST.CLASSIFIER.feature_index(i);
    a = WEAK_NEW.(WEAK_NEW.list{f_ind,1})(WEAK_NEW.list{f_ind,2});
    CTEST.CLASSIFIER.haar(i).descriptor = a.descriptor;
    CTEST.CLASSIFIER.haar(i).hinds = a.hinds;
    CTEST.CLASSIFIER.haar(i).hvals = a.hvals;
    CTEST.CLASSIFIER.haar(i).polarity = CASCADE(1).CLASSIFIER.polarity(i);
    CTEST.CLASSIFIER.haar(i).theta    = CASCADE(1).CLASSIFIER.theta(i);
end


