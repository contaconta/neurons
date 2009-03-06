function SET = ada_cascade_precom(SET, CASCADE, LEARNERS, FILES)


learner_list = ada_learner_list(CASCADE);  NLEARNERS = length(learner_list);
types = extract_types(learner_list);  edgems = extract_edgemethods(learner_list); %angles = extract_angles(learner_list);
hog_stuff = extract_hog_stuff(LEARNERS);

NIMAGES = 100000; %size(SET.Images,3);  
j = 1;

%MEMORYLIMIT = 2000000000;

disp('   beginning precomputing');

W = wristwatch('start', 'end', NIMAGES, 'every', 1000, 'text', '    ...precomputed example ');
%block = min(NIMAGES, round(FILES.memory / (NLEARNERS*SET.responses.bytes)));
block = 50000;
A = single(zeros(block, NLEARNERS));

for e = 1:NIMAGES
    
    %I = imread('/osshare/Work/Data/nuclei24/train/pos/nuclei00001.png');
    A(j,:) = extract_features(SET.Images(:,:,j), learner_list, LEARNERS, types, edgems, hog_stuff);

    W = wristwatch(W, 'update', e); %#ok<NASGU>
    j = j + 1;
    if j > block; j = 1;   disp('saving A'); save A.mat A; end;
end


    


%% extracts the features belonging to example image I
function flist = extract_features(I, learner_list, LEARNERS, types, edgems, hog_stuff)

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



% now, we can use the precomputed features to fill in needed features of flist
for l = 1:length(learner_list)
    switch learner_list{l}.type
        case 'intmean'
            flist(l) = ada_intmean_response(I);
        case 'intvar'
            flist(l) = ada_intvar_response(I);
        case 'haar'
            %hinds = learner_list{l}.hinds;
            %hvals = learner_list{l}.hvals;
            flist(l) = ada_haar_response(learner_list{l}.hinds, learner_list{l}.hvals, II);
        case 'hog'
            %oind = learner_list{l}.oind;
            %cellr = learner_list{l}.cellr;
            %cellc = learner_list{l}.cellc;
            %n = learner_list{l}.n;
            
            flist(l) = HOG(learner_list{l}.cellr, learner_list{l}.cellc, learner_list{l}.oind, learner_list{l}.n);
            %flist(l) = ada_hog_response(I, oind, cellc, cellr, n, bins, cellsize, blocksize, HOG);
        case 'spedge'
            %angle = learner_list{l}.angle;
            %stride = learner_list{l}.stride;
            edge_method = learner_list{l}.edge_method;
            %row = learner_list{l}.row;
            %col = learner_list{l}.col;
                
            % if we've already computed EDGE for SIGMA, use it, otherwise use I
            ind = find([edgelist(:).edge_method] == edge_method,1);
               
            %EDGE = edgelist(ind).EDGE;
            flist(l) = single_spedge(learner_list{l}.angle, learner_list{l}.stride, edge_method, learner_list{l}.row, learner_list{l}.col, edgelist(ind).EDGE, 'edge');
            
        case 'spdiff'
            %angle1 = learner_list{l}.angle1;
            %angle2 = learner_list{l}.angle2;
            %stride = learner_list{l}.stride;
            edge_method = learner_list{l}.edge_method;
            %row = learner_list{l}.row;
            %col = learner_list{l}.col;
            
            % if we've already computed EDGE for SIGMA, use it, otherwise use I
            ind = find([edgelist(:).edge_method] == edge_method,1);
               
            %EDGE = edgelist(ind).EDGE;
            flist(l) = ada_spdiff_response(learner_list{l}.angle1,learner_list{l}.angle2,learner_list{l}.stride, edge_method, learner_list{l}.row,learner_list{l}.col, edgelist(ind).EDGE, 'edge');
            %flist(l) = single_spedge(learner_list{l}.angle, learner_list{l}.stride, edge_method, learner_list{l}.row, learner_list{l}.col, edgelist(ind).EDGE, 'edge');
            
                
        case 'spangle'
            %angle = learner_list{l}.angle;
            %stride = learner_list{l}.stride;
            edge_method = learner_list{l}.edge_method;
            %row = learner_list{l}.row;
            %col = learner_list{l}.col;
                
            % if we've already computed EDGE for SIGMA, use it, otherwise use I
            ind = find([edgelist(:).edge_method] == edge_method,1);
            flist(l) = single_spangle(learner_list{l}.angle, learner_list{l}.stride, edge_method, learner_list{l}.row, learner_list{l}.col, edgelist(ind).EDGE, G, gh, gv, 'edge');
                   
        case 'spnorm'
            %angle = learner_list{l}.angle;
            %stride = learner_list{l}.stride;
            edge_method = learner_list{l}.edge_method;
            %row = learner_list{l}.row;
            %col = learner_list{l}.col;
                
            % if we've already computed EDGE for SIGMA, use it, otherwise use I
            ind = find([edgelist(:).edge_method] == edge_method,1);
            flist(l) = single_spnorm(learner_list{l}.angle, learner_list{l}.stride, edge_method, learner_list{l}.row, learner_list{l}.col, edgelist(ind).EDGE, G, gh, gv, 'edge');
                  
    end
end

%flist = rand([1 length(learner_list)]);

% spokes_present = 0;
% for t = 1:length(types)
%     
%     switch types(t)
%         case 'intmean'
%             
%         case 'intvar'
%             
%         case 'haar'
%             II = integral_image(I);
%         case 'hog'
%             
%         otherwise  % we've got spedges, spangles, etc..
%             if spokes_present == 0
%                 for e = 1:length(edgems)
%                     edgelist(e) = edgemethods(edgems(e));
%                 end
%                 spokes_present = 1;
%             end
%             if strcmp(types(t),'spnorm') || strcmp(types(t), 'spangle')
%                 gh = imfilter(I,fspecial('sobel')' /8,'replicate');
%                 gv = imfilter(I,fspecial('sobel')/8,'replicate');
%                 G(:,:,1) = gv;
%                 G(:,:,2) = gh;
%                 G = gradientnorm(G);
%     end
% end

%[f EDGE] = single_spedge(0,2, 59,3, 4, I);

%II = integral_image(I);
%sp = spedges(I, 0:30:360-30, 2, 59);
%E = edgemethods(I, 59);
                

%flist = rand([1 length(learner_list)]);


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
