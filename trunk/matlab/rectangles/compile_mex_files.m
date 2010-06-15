


%% COMPILE ANY MISSING MEX FILES

if ~exist(['weight_sample_mex.' mexext], 'file')
    mex -O weight_sample_mex.cpp
end


if ~exist(['ChooseThreshold_mex.' mexext], 'file')
    mex -v -O ChooseThreshold_mex.cpp
end


if ~exist(['AdaBoostClassifyDynamicA_mex.' mexext], 'file');
    mex -v -O AdaBoostClassifyDynamicA_mex.cpp
end