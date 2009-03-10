function [A, SET] = ada_cascade_precom(SET, CASCADE, LEARNERS, filenm, varargin)


learner_list = ada_learner_list(CASCADE);  NLEARNERS = length(learner_list);
types = extract_types(learner_list);  edgems = extract_edgemethods(learner_list); %angles = extract_angles(learner_list);
hog_stuff = extract_hog_stuff(LEARNERS);

NIMAGES = size(SET.Images,3);  
j = 1;

load LN24.mat;

disp('    beginning precomputing');
block = min(NIMAGES, 100000);

W = wristwatch('start', 'end', block, 'every', 1000, 'text', '    ...precomputed example ');

% we will store it as a double unless varargin{1} = 'single'
if nargin > 4
    if strcmp(varargin{1}, 'single')
        disp('using single precision');
        A = single(ones(block, NLEARNERS));
    end
else
    disp('using double precision');
   % A = ones(block, NLEARNERS);
    A = rand(block, NLEARNERS);
end

        

for e = 1:NIMAGES
    
    A(j,:) = extract_features(SET.Images(:,:,j), learner_list, LEARNERS, types, edgems, hog_stuff, LN);

    W = wristwatch(W, 'update', e); %#ok<NASGU>
    j = j + 1;
end

save(filenm, 'A');





%% =================== SUPPORTING FUNCTIONS =====================



%% extracts the features belonging to example image I
function flist = extract_features(I, learner_list, LEARNERS, types, edgems, hog_stuff, LN)

flist = zeros(size(learner_list));

% first, we should precompute things that only need to be computed once
if ~isempty(find(strcmp(types, 'haar'),1))
    II = integral_image(I);
    II = II(:);
end

if ~isempty(find(strcmp(types, 'hog'),1))
    bins = hog_stuff.bins;
    cellsize = hog_stuff.cellsize;
    blocksize = hog_stuff.blocksize;
    [f, HOG] = ada_hog_response(I, 1, 1, 1, 1, bins, cellsize, blocksize);
end

if ~isempty(find(strcmp(types, 'spedge'),1)) || ~isempty(find(strcmp(types, 'spdiff'),1)) || ~isempty(find(strcmp(types, 'spangle'),1)) || ~isempty(find(strcmp(types, 'spnorm'),1))
    for e = 1:length(edgems)
        edgelist(e).EDGE = edgemethods(I, edgems(e));
        edgelist(e).edge_method = edgems(e);
    end
end

if ~isempty(find(strcmp(types, 'spangle'),1)) || ~isempty(find(strcmp(types, 'spnorm'),1))
    gh = imfilter(I,fspecial('sobel')' /8,'replicate');
    gv = imfilter(I,fspecial('sobel')/8,'replicate');
    G(:,:,1) = gv;
    G(:,:,2) = gh;
    G = gradientnorm(G);
end



% %=============== DEBUG FOR PERSONS =============
% LN = [];
% %==============================================

% now, we can use the precomputed features to fill in needed features of flist
for l = 1:length(learner_list)
    switch learner_list{l}.type
        case 'intmean'
            flist(l) = ada_intmean_response(I);
        case 'intvar'
            flist(l) = ada_intvar_response(I);
        case 'haar'
            flist(l) = ada_haar_response(learner_list{l}.hinds, learner_list{l}.hvals, II);
        case 'hog'
            flist(l) = HOG(learner_list{l}.cellr, learner_list{l}.cellc, learner_list{l}.oind, learner_list{l}.n);
        case 'spedge'
            edge_method = learner_list{l}.edge_method;
            ind = find([edgelist(:).edge_method] == edge_method,1);
            flist(l) = single_spedge(learner_list{l}.angle, learner_list{l}.stride, edge_method, learner_list{l}.row, learner_list{l}.col, LN, edgelist(ind).EDGE, 'edge');
        case 'spdiff'
            edge_method = learner_list{l}.edge_method;
            ind = find([edgelist(:).edge_method] == edge_method,1);
            flist(l) = ada_spdiff_response(learner_list{l}.angle1,learner_list{l}.angle2,learner_list{l}.stride, edge_method, learner_list{l}.row,learner_list{l}.col, LN, edgelist(ind).EDGE, 'edge');
        case 'spangle'
            edge_method = learner_list{l}.edge_method;
            ind = find([edgelist(:).edge_method] == edge_method,1);
            flist(l) = single_spangle(learner_list{l}.angle, learner_list{l}.stride, edge_method, learner_list{l}.row, learner_list{l}.col,  LN, edgelist(ind).EDGE, G, gh, gv,  'edge');     
        case 'spnorm'
            edge_method = learner_list{l}.edge_method;
            ind = find([edgelist(:).edge_method] == edge_method,1);
            flist(l) = single_spnorm(learner_list{l}.angle, learner_list{l}.stride, edge_method, learner_list{l}.row, learner_list{l}.col, LN, edgelist(ind).EDGE, G, gh, gv,  'edge');
    end
end







%% ====================SUPPORTING FUNCTIONS=============================


function hog_stuff = extract_hog_stuff(LEARNERS)
    hog_stuff = [];

    for l = 1:length(LEARNERS)
        if strcmp(LEARNERS(l).feature_type, 'hog')
            
            hog_stuff.bins = LEARNERS(l).bins;
            hog_stuff.cellsize = LEARNERS(l).cellsize;
            hog_stuff.blocksize = LEARNERS(l).blocksize;
        end
    end



function types = extract_types(learner_list)

types = {};

for i = 1:length(learner_list)
    types = union(types, learner_list{i}.type);
end


function edgems = extract_edgemethods(learner_list)

edgems = [];

for i = 1:length(learner_list)
    if isfield(learner_list{i}, 'edge_method')
        edgems = union(edgems, learner_list{i}.edge_method);
    end
end


function angles = extract_angles(learner_list)

angles = [];

for i = 1:length(learner_list)
    if isfield(learner_list{i}, 'angle')
        angles = union(angles, learner_list{i}.angle);
    end
    if isfield(learner_list{i}, 'angle1')
        angles = union(angles, learner_list{i}.angle1);
    end
    if isfield(learner_list{i}, 'angle2')
        angles = union(angles, learner_list{i}.angle2);
    end

end

% angle1 angle2 spdiff
% angle spedge
% angle spangle
% angle spnorm
