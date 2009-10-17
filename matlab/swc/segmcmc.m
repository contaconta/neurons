%% SEGMCMC performs SWC MCMC segmentation on mitochondria images
%

%   Copyright © 2009 Computer Vision Lab, 
%   École Polytechnique Fédérale de Lausanne (EPFL), Switzerland.
%   All rights reserved.
%
%   Authors:    Kevin Smith         http://cvlab.epfl.ch/~ksmith/
%               Aurelien Lucchi     http://cvlab.epfl.ch/~lucchi/
%
%   This program is free software; you can redistribute it and/or modify it 
%   under the terms of the GNU General Public License version 2 (or higher) 
%   as published by the Free Software Foundation.
%                                                                     
% 	This program is distributed WITHOUT ANY WARRANTY; without even the 
%   implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
%   PURPOSE.  See the GNU General Public License for more details.



%% =========================== PARAMETERS =================================

S = 30000;                  % number of samples in MCMC
labelList = [1 2];          % classification labels [1=bg 2=boundary 3=mito interior]
kernelType = 2;             % RBF kernel
WINDOW = [1 480 1 640];     % region of the image we will work on [r1 r2 c1 c2]
T = 1.5;                    % temperature

initFlag = 1;               % performs initialization functions, bypass by setting to 0 
displayFlag = 1;            % displays what the segmentation is doing
updateEvery = 100;          % how often should we update the display?

modelCell = {'perfect'};    % specifies which classification model to use


%% ====================== INITIALIZATION STUFF ============================

if initFlag
    
    % set necessary paths
    addpath([pwd '/utils/']); %#ok<UNRCH>
    addpath([pwd '/bin/']);
    addpath([pwd '/utils/libsvm-mat-2.89-3/']);
    huttPath = '/osshare/Work/software/huttenlocher_segment/';
    imPath = [pwd '/images/'];

    % fix the random number stream
    st = RandStream.create('mt19937ar','seed',5489);  RandStream.setDefaultStream(st);  %rand('twister', 100);    % seed for Matlab 7.8 (?)

    % load the image we are working with
    Iraw = imread('images/FIBSLICE0002.png');

    % load the ground truth segmentation
    Igt = imread('images/annotation0002.png');  Igt = Igt > 0;
     

    % load superpixels or atomic regions as a label matrix, L
    disp('Reading the superpixel segmentation from file');
    L = readRKLabel('temp/labels/FIBSLICE0002.dat', [1536 2048])';

    % work on a sub-region of the image to keep things fast
    Iraw = Iraw(WINDOW(1):WINDOW(2), WINDOW(3):WINDOW(4));
    L = L(WINDOW(1):WINDOW(2), WINDOW(3):WINDOW(4));
    Igt = Igt(WINDOW(1):WINDOW(2), WINDOW(3):WINDOW(4));

    % rename the labels since we have cropped the image
    disp('Re-labeling the superpixel segmentation after cropping');
    Llist = unique(L);
    for l = 1:length(Llist)
        L(L == Llist(l)) = l;
    end

    % create useful images showing superpixel regions
    disp('Forming superpixel images Ig and Is');
    Ig = label2gray(L,Iraw); Ig = uint8(round(Ig));
    Is = imseg(Iraw, L);

    % create the initial fully connected adjacency graph G0
    disp('Extracting adjacency graph G0 from superpixel segmentation image.');
    [G0, G0list] = adjacency(L);
    
    
    % create a list of superpixel center locations and pixel lists
    disp('Computing superpixel center locations and GT labels.');
    centers = zeros(max(L(:)),1); pixelList = cell([1 size(G0,1)]);
    for l = 1:max(L(:))
        pixelList{l} = find(L == l); 
        [r,c] = ind2sub(size(L), pixelList{l});
        centers(l,1) = mean(r); centers(l,2) = mean(c);
        GT(l) = median(double(Igt(pixelList{l}))) + 1;
    end
    
    % precompute the KL divergences for each edge
    disp('Precomputing the KL divergences.');
    KL = edgeKL(Iraw, pixelList, G0, 1);
      
%     % initialize the SVM model
%     if useGroundTruth==false
%       if ~exist('model', 'var')
%         disp('Computing the SVM model.');
%         feature_vectors = [pwd '/temp/Model-0-4200-3-sup/feature_vectors'];
%         [label_vector, instance_matrix] = libsvmread(feature_vectors);
%         training_label = label_vector(1:4000,:);
%         training_instance = instance_matrix(1:4000,:);
% 
%         [model,minI,maxI] = loadModel(training_label,training_instance, ...
%                                       rescaleData,kernelType);
%       end
%     else
%       disp('No model computed. Ground truth data will be used.');
%       model = 0;
%       minI = 0;
%       maxI = 0;
%     end
   
    %create an initial partition of the graph
    disp('Creating a random initial partiton W of the graph.');
    B1 = .5;
    W = swc_CP1(G0, B1, KL);         % W = adjacency matrix with initial cuts
    
  	% find the cliques (Cw contains the 'clique color' of each node)
    disp('Finding the cliques');
    [numCw,Cw] = graphconncomp(W, 'directed', 0);   % assign the colors
    W = swc_AdjFromColor(G0,Cw);                    % fill in missing edges according to colors
    
    
    % assign classification labels to the initial partition
    disp('Randomly assigning labels to each region in the graph.')
    LABELS = zeros(size(Cw));
    for c = 1:numCw
        members = find(Cw == c)';   % find the clique members

        % do a random assignment to the clique
        LABELS(members) = randsample(labelList,1);
        if rand(1) < .5
            LABELS(members) = 1;
        else
            LABELS(members) = 2;
        end
    end
    
    % plot the initial partition
    figure(1234); 
    gplotl(W,centers,LABELS,Is); set(gcf, 'Position', [300 20, 1200 900]);
   	pause(0.06); refresh;
end




%% ======================== PRE-SEGMENTATION ==============================

% set the annealing schedule
Bstart = 10;  Bend = .2;
B = Bstart*ones([1 S]); %B = [Bstart linterp([1 S], [Bstart Bend], 2:S-1) Bend];

if strcmp(modelCell{1}, 'perfect')
    modelCell{2} = GT;
end

% compute the initial posterior (log posterior)
disp('Computing initial posterior');
P = zeros([1 S]);
%Plist = swc_post2(W, LABELS, modelCell, pixelList, Iraw, [], 'init');
Plist = swc_post2(W, LABELS, GT, pixelList, Iraw, [], 'init');
P(1) = sum(Plist);



%% ================= Metropolis-Hastings SEGMENTATION =====================
disp('Applying metropolis-hastings with Swendson-Wang cuts.')


for s = 2:S
    
    % step 1: select a seed vertex v
    %v = randsample(size(W,1),1);
    v = randsample(size(W,1),1, true, -1*(Plist - max(Plist)-.2));  % select v's with poor likelihood at higher probability
    
    % step 2: grow a region V0 with color V0c
    B_CUT = .6;
    [V0, V0a] = swc_swc2_1(W, B_CUT, v, KL);  	% get a list of the members of V0
    V0c = Cw(V0(1));                    % the current color of V0
    %disp(['V0 size=' num2str(size(V0,2)) ', NbAdj=' num2str(sum(sum(V0a==1)))])

    % determine what are the neighbors colors
    [neighbors,junk] = find(G0(:,V0));
    neighbors = unique(setdiff(neighbors, V0));
    neighborColors = unique(Cw(neighbors));
    neighborColors = setdiff(neighborColors, V0c);
    
    
    % step 3: choose a new color & label for V0
    newColor = ones(1,max(1,size(neighborColors,2))).*(max(Cw)+1);
    c = randsample([neighborColors newColor], 1);  % c is the color assigned to V0
    if c == max(Cw)+1
        newL = randsample(labelList, 1);
        type_move = 'Split';
    else
        newL = LABELS(find(Cw == c, 1));
        type_move = 'Merge';
    end
    
    % step 4: create a proposal copy of LABELS
    LABELSp = LABELS;
    LABELSp(V0) = newL;
    
    % step 5: compute the acceptance ratio (log likelihood)
    Cwp = Cw;           % proposed colors
    Cwp(V0) = c;        % assign c to V0
    
    % affected regions include cliques with color V0c and with color c
    R = find(Cw == V0c);
    R = [R find(Cw == c)]; %#ok<AGROW>
    
    % update likelihood values in the proposal
    %Pplist = swc_post2(W, LABELSp, modelCell, pixelList, Iraw, R, 'init');
    %Pplist1 = swc_post2(W, LABELSp, GT, pixelList, Iraw, R, 'init');
    Pplist = swc_post2(W, LABELSp, GT, pixelList, Iraw, R, Plist);

%     if ~isempty(find(Pplist ~= Pplist1))
%         keyboard;
%     end
    
    % compute posterior terms for previous and proposal
    Pp = sum(Pplist(V0));
    Pold = sum(Plist(V0));
    
    % acceptance ratio
    a = exp( B(s)* (Pp - Pold));
    
    
    % step 6: accept or reject ( W -> W or W -> Wp)
    r = rand(1);
    if r <= a
        % ACCEPT THE PROPOSAL
        
      	Cw(V0) = c;         % apply new color c to V0
        W = swc_AdjFromColor(G0, Cw, W, [V0c c neighborColors]);
        [numCw,Cw] = graphconncomp(W, 'directed', 0);
        LABELS = LABELSp;
        Plist = Pplist;
        P(s) = sum(Pplist);
        %disp(['accepted sample ' num2str(s) ', a=' num2str(a) ', newL=' num2str(newL)]);
        
    else
        % REJECT THE PROPOSAL
        
         P(s) = P(s-1);
         %disp(['rejected sample ' num2str(s) ', a=' num2str(a) ', L=' num2str(newL)]);
    end
    
    if displayFlag && (mod(s,updateEvery) == 0)
        figure(1234); cla;
        gplotl(W,centers,LABELS,Is); %set(gcf, 'Position', [300 20, 1200 900]);
        pause(0.05); refresh;
        
        figure(445); cla;
        plot(P(1:s)); grid on;
        
        pause(0.05); refresh;
    end
    
end