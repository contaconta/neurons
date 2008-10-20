function WEAK = ada_define_weak_classifiers(varargin)
%ADA_DEFINE_CLASSIFIERS defines a set of weak haar like classifiers.
%
%   WEAK = ada_define_classifiers(IMSIZE, ...) constructs a set of weak 
%
%   Copyright 2008 Kevin Smith
%
%   See also ADA_PLOT_HAAR_FEATURE


%% handle input parameters
IMSIZE = varargin{1};
for i = 2:nargin
    if strcmp(varargin{i}, 'type')
        TYPES = varargin{i+1};
    end
end


%% initialize the learners structure to be empty
WEAK.learners = [];

% add haar type 1
haars = ada_define_haar_wavelets(IMSIZE, 'type', 'haar1');
WEAK.learners(length(WEAK.learners)+1:length(WEAK.learners)+length(haars)) = haars(:);
clear haars;

% add haar type 2
haars = ada_define_haar_wavelets(IMSIZE, 'type', 'haar2');
WEAK.learners(length(WEAK.learners)+1:length(WEAK.learners)+length(haars)) = haars(:);
clear haars;

% add haar type 3
haars = ada_define_haar_wavelets(IMSIZE, 'type', 'haar3');
WEAK.learners(length(WEAK.learners)+1:length(WEAK.learners)+length(haars)) = haars(:);
clear haars;

% add haar type 5
haars = ada_define_haar_wavelets(IMSIZE, 'type', 'haar4');
WEAK.learners(length(WEAK.learners)+1:length(WEAK.learners)+length(haars)) = haars(:);
clear haars;


