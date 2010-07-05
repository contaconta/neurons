%% COMPILE ANY MISSING MEX FILES

if ~exist(['weight_sample_mex.' mexext], 'file')
    mex -v -O weight_sample_mex.cpp
end
if ~exist(['ChooseThreshold_mex.' mexext], 'file')
    mex -v -O ChooseThreshold_mex.cpp
end
if ~exist(['AdaBoostClassify_mex.' mexext], 'file')
    mex -v -O AdaBoostClassify_mex.cpp
end