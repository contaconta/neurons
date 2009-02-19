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
    
     %% add intmean weak learners
    if strcmp('intmean', LEARNERS(t).feature_type)
        
        intmean = ada_intmean_define(LEARNERS(t).IMSIZE );
        WEAK.lists.intmean = [];
        
        for i = 1:length(intmean)
            WEAK.learners{length(WEAK.learners)+1} = intmean(i);
            WEAK.learners{length(WEAK.learners)}.type = 'intmean';
        end
        disp(['   added ' num2str(length(intmean)) ' ' LEARNERS(t).feature_type ' learners']);
        clear intmean;
    end
    
    %% add invar weak learners
    if strcmp('intvar', LEARNERS(t).feature_type)
        
        intvar = ada_intvar_define(LEARNERS(t).IMSIZE);
        WEAK.lists.intvar = [];
        
        for i = 1:length(intvar)
            WEAK.learners{length(WEAK.learners)+1} = intvar(i);
            WEAK.learners{length(WEAK.learners)}.type = 'intvar';
        end
        disp(['   added ' num2str(length(intvar)) ' ' LEARNERS(t).feature_type ' learners']);
        clear intvar;
    end
    
    %% add haar wavelet weak learners
    if strcmp('haar', LEARNERS(t).feature_type)
        
        % call the function to define haar wavelets and pass them to WEAK
        haars = ada_haar_define(LEARNERS(t).IMSIZE, 'shapes', LEARNERS(t).shapes, 'SCAN_Y_STEP', LEARNERS(t).SCAN_Y_STEP, 'SCAN_X_STEP', LEARNERS(t).SCAN_X_STEP);
        WEAK.lists.haar = [];
        
        for i = 1:length(haars)
            WEAK.learners{length(WEAK.learners)+1} = haars(i);
            WEAK.learners{length(WEAK.learners)}.type = 'haar';
        end
        disp(['   added ' num2str(length(haars)) ' ' LEARNERS(t).feature_type ' learners']);
        clear haars;   
    end

    
    %% add spedge weak learners
    if strcmp('spedge', LEARNERS(t).feature_type)
        
        spedge = ada_spedge_define(LEARNERS(t).IMSIZE, LEARNERS(t).angles, LEARNERS(t).stride, LEARNERS(t).edge_methods);
        WEAK.lists.spedge = [];
        
        for i = 1:length(spedge)
            WEAK.learners{length(WEAK.learners)+1} = spedge(i);
            WEAK.learners{length(WEAK.learners)}.type = 'spedge';
        end
        disp(['   added ' num2str(length(spedge)) ' ' LEARNERS(t).feature_type ' learners']);
        clear spedge;
    end
    
    %% add spdiff weak learners
    if strcmp('spdiff', LEARNERS(t).feature_type)
        
        spdiff = ada_spdiff_define(LEARNERS(t).IMSIZE, LEARNERS(t).angles, LEARNERS(t).stride, LEARNERS(t).edge_methods);
        WEAK.lists.spdiff = [];
        
        for i = 1:length(spdiff)
            WEAK.learners{length(WEAK.learners)+1} = spdiff(i);
            WEAK.learners{length(WEAK.learners)}.type = 'spdiff';
        end
        disp(['   added ' num2str(length(spdiff)) ' ' LEARNERS(t).feature_type ' learners']);
        clear spdiff;
    end
    
    %% add hog weak learners
    if strcmp('hog', LEARNERS(t).feature_type)
        
        hog = ada_hog_define(LEARNERS(t).IMSIZE, LEARNERS(t).bins, LEARNERS(t).cellsize, LEARNERS(t).blocksize);
        WEAK.lists.hog = [];
        
        for i = 1:length(hog)
            WEAK.learners{length(WEAK.learners)+1} = hog(i);
            WEAK.learners{length(WEAK.learners)}.type = 'hog';
        end
        disp(['   added ' num2str(length(hog)) ' ' LEARNERS(t).feature_type ' learners']);
        clear hog;
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



