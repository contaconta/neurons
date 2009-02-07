function WEAK = ada_define_weak_learners(LEARNERS)
%ADA_DEFINE_LEARNERS defines a set of weak haar like classifiers.
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


%% define the general parameters of a weak learner
WEAK.IMSIZE = LEARNERS(1).IMSIZE;
WEAK.error = [];
WEAK.learners = {};
WEAK.lists = [];

for t = 1:length(LEARNERS)
    %% add haar wavelet weak learners
    if strcmp('haar', LEARNERS(t).feature_type)
        
        % call the function to define haar wavelets and pass them to WEAK
        haars = ada_haar_define(LEARNERS(t).IMSIZE, 'shapes', LEARNERS(t).shapes);
        WEAK.lists.haar = [];
        
        for i = 1:length(haars)
            WEAK.learners{length(WEAK.learners)+1} = haars(i);
            WEAK.learners{length(WEAK.learners)}.type = 'haar';
        end
        clear haars;   
    end

    
    %% add spedge weak learners
    if strcmp('spedge', LEARNERS(t).feature_type)
        
        spedge = ada_spedge_define(LEARNERS(t).IMSIZE, LEARNERS(t).angles, LEARNERS(t).sigma);
        WEAK.lists.spedge = [];
        
        for i = 1:length(spedge)
            WEAK.learners{length(WEAK.learners)+1} = spedge(i);
            WEAK.learners{length(WEAK.learners)}.type = 'spedge';
        end
        clear spedge;
    end
end

WEAK.error = zeros(length(WEAK.learners),1);
WEAK = orderfields(WEAK);



% make some lists of each of the learners
for l = 1:length(WEAK.learners)
    type = WEAK.learners{l}.type;
    
    WEAK.lists.(type) = [WEAK.lists.(type) l];
end
    


%% ===== namenewfield ===========================================================
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



%% handle input parameters
% IMSIZE = varargin{1};
% LEARNERS 
% TYPES = varargin{2};
% PARAMS = varargin{3};


 %WEAK.error              = [WEAK.error; zeros(length(WEAK.(field)),1)];

        % set the feature definitions to a field of WEAK('haars', 'haars_1', etc)
        %field = namenewfield(WEAK, 'haars');
        %WEAK.(field) = haars;

         
        %index_map               = single(sort(length(WEAK.error):-1:length(WEAK.error)-length(WEAK.(field))+1));
        %WEAK.list(index_map,1)   = repmat({field}, [length(WEAK.(field)),1]);
        %WEAK.list(index_map,2)   = num2cell( single(1:length(WEAK.(field))))';
        %WEAK.list(index_map,3)   = num2cell( single(repmat(t, [length(WEAK.(field)),1])));
        %WEAK.learners   = {WEAK.learners{:}, {'haar', field, index_map, @ada_haar_learn, @ada_haar_response, @ada_haar_trclassify1, @ada_haar_trclassify2}};
        


  %WEAK.error              = [WEAK.error; zeros(length(WEAK.(field)),1)]; 
        
%         field = namenewfield(WEAK, 'spedge');
%         WEAK.(field) = spedge;
%         index_map               = single(sort(length(WEAK.error):-1:length(WEAK.error)-length(WEAK.(field))+1));
%         WEAK.list(index_map,1)   = repmat({field}, [length(WEAK.(field)),1]);
%         WEAK.list(index_map,2)   = num2cell( single(1:length(WEAK.(field))) )';
%         WEAK.list(index_map,3)   = num2cell( single(repmat(t, [length(WEAK.(field)),1])));
%         WEAK.learners   = {WEAK.learners{:}, {'spedge', field, index_map, @ada_spedge_learn, @ada_spedge_response, @ada_spedge_trclassify1, @ada_spedge_trclassify2}};
        

