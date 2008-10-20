function CLASSIFIER = ada_classifier_init(T, WEAK)
%
%
%
%
%
%


CLASSIFIER.feature_index =          zeros(1,T);
%CLASSIFIER.feature_descriptor =     zeros(T, size(WEAK.descriptor, 2));
%CLASSIFIER.fast =                   zeros(T, size(WEAK.fast, 2));
%CLASSIFIER.polarity =               zeros(1,T);
%CLASSIFIER.theta =                  zeros(1,T);
CLASSIFIER.alpha =                  zeros(1,T);
CLASSIFIER.w =                      [];
CLASSIFIER.IMSIZE =                 WEAK.IMSIZE;
CLASSIFIER.type =                   'SINGLE';  % specify if this is a cascade or single classifier
CLASSIFIER.learner_type             = {};
CLASSIFIER.functions                = {};