function WEAK = ada_define_weak_classifiers(varargin)
%ADA_DEFINE_CLASSIFIERS defines a set of weak haar like classifiers.
%
%   WEAK = ada_define_classifiers(IMSIZE, ...) constructs a set of weak 
%
%   WEAK = ada_define_weak_learners(IMSIZE, {'haar', 'spedge'}, ...
%               { {IMSIZE, 'type', {'haar1', 'haar2', 'haar3','haar5'}}, ...
%               {IMSIZE, 'whatever'} });
%   
%   Copyright 2008 Kevin Smith
%
%   See also ADA_PLOT_HAAR_FEATURE, ADA_DEFINE_HAAR_WAVELETS


%% handle input parameters
IMSIZE = varargin{1};
TYPES = varargin{2};
PARAMS = varargin{3};

%% define the general parameters of a weak learner
WEAK.IMSIZE = IMSIZE;
WEAK.error = [];
WEAK.polarity = [];
WEAK.theta = [];
WEAK.learners = {};

for t = 1:length(TYPES)
    %% add haar wavelet weak learners
    if strcmp('haar', TYPES(t))
        % get the passed parameters
        params = PARAMS{t}; %#ok<FNDSB>

        % call the function to define haar wavelets and pass them to WEAK
        haars = ada_haar_define(params{:});

        % set the wavelet definitions to a field ('haars', 'haars_1', etc)
        field = namenewfield(WEAK, 'haars');
        WEAK.(field) = haars;

        WEAK.error              = [WEAK.error; zeros(length(WEAK.(field)),1)]; 
        index_map               = sort(length(WEAK.error):-1:length(WEAK.error)-length(WEAK.(field))+1);
        WEAK.list(index_map,1)   = repmat({field}, [length(WEAK.(field)),1]);
        WEAK.list(index_map,2)   = num2cell(1:length(WEAK.(field)))';
        WEAK.list(index_map,3)   = num2cell( repmat(t, [length(WEAK.(field)),1]));
        WEAK.learners   = {WEAK.learners{:}, {'haar', field, index_map, @ada_haar_learn, @ada_haar_response, @ada_haar_classify}};
        clear haars;
        
        WEAK = orderfields(WEAK);
    end

    %% add spedge weak learners
    if strcmp('spedge', TYPES(t))


    end
end






%% namenewfield
function newfield = namenewfield(STRUCT, field)

if isfield(STRUCT, field)
    if strfind(field, '_')
        num = str2double(field(strfind(field,'_')+1:length(field)));
        base = strtok(field, '_');
        newfield = strcat(base, '_', num2str(num + 1));
        newfield = namenewfield(STRUCT,newfield);
    else
        newfield = strcat(field, '_1');
        newfield = namenewfield(STRUCT,newfield);
    end
else
    newfield = field;
end
