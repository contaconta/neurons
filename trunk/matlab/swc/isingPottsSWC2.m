% ========= parameters ==================
S = 10000;

% set necessary paths
addpath([pwd '/utils/']);
addpath([pwd '/bin/']);
huttPath = '/osshare/Work/software/huttenlocher_segment/';
imPath = [pwd '/images/'];

% fix the random number stream
s = RandStream.create('mt19937ar','seed',5489);  RandStream.setDefaultStream(s);  %rand('twister', 100);    % seed for Matlab 7.8 (?)

% load an image we want to play with
Iraw = imread([imPath 'test.png']);

% load superpixels or atomic regions as a label matrix, L
HUT = imread([imPath 'testHUTT.ppm']);
L = rgb2label(HUT);

% G contains average gray levels of I for regions in L 
Ig = label2gray(L,Iraw); Ig = uint8(round(Ig));

% extract and adjacency matrix and list from L
[G0, G0list] = adjacency(L);

% create a list of superpixel center locations
centers = zeros(max(L(:)),1);
for l = 1:max(L(:))
    pixelList = find(L == l); 
    [r,c] = ind2sub(size(L), pixelList);
    centers(l,1) = mean(r);
    centers(l,2) = mean(c);
end

% % plot the original image
% figure; imshow(Iraw);  axis image off; set(gca, 'Position', [0 0 1 1]);
% 
% % plot the segmentation with average gray levels
% figure; imshow(Ig);  axis image off; set(gca, 'Position', [0 0 1 1]);
% 
% % plot the superpixel centers
% figure; imshow(Ig);  axis image off; set(gca, 'Position', [0 0 1 1]);
% hold on; plot(centers(:,2), centers(:,1), 'b.');

% % plot the adjacency graph
% figure; imshow(Iraw); axis image off; set(gca, 'Position', [0 0 1 1]);
% hold on; gplot(G0, [centers(:,2) centers(:,1)], '.-'); 


B1 = .25;
%create an initial partition of the graph
W = swc_CP(G0, B1);                              % make initial cuts
[numCw,Cw] = graphconncomp(W, 'directed', 0);   % color the graph
W = swc_AdjFromColor(G0,Cw);                    % fill in missing adjacency edges

% show the initial partition
% figure; imshow(Iraw); axis image off; set(gca, 'Position', [0 0 1 1]);
% hold on; gplot(W, [centers(:,2) centers(:,1)], '.-');


% assign colors & labels to the initial partition
LABELS = zeros(size(Cw));
for c = 1:numCw
    members = find(Cw == c)';
    if rand(1) < .5
        LABELS(members) = 1;
    else
        LABELS(members) = 2;
    end
end


% % show the initial partition
% figure;
% gplotl(W,centers,LABELS,Iraw);
% figure;
% gplotc(W, centers, Cw, Iraw);


T = 1.5;  
Bstart = .66;  Bend = .2;
B = [Bstart linterp([1 S], [Bstart Bend], 2:S-1) Bend];

%% ===================== Metropolis-Hastings ============================


% compute the initial posterior
P = zeros([1 S]);
P(1) = pottsPost(G0, LABELS, T);


%% build the markov chain
for s = 2:S
    
    % step 1: select a seed vertex v
    v = randsample(size(W,1),1);
    
    % step 2: grow a region V0 with color
    [V0, V0a] = swc_swc2(W, B(s), v);  	% get a list of the members of V0
    V0c = Cw(V0(1));                    % the current color of V0

    % determine what are the neighbors colors
    neighbors = [];
    for m = V0
        neighbors = [neighbors graphtraverse(G0,m, 'directed', 0, 'depth', 1)]; %#ok<AGROW>
    end
    neighbors = unique(setdiff(neighbors, V0));
    neighborColors = unique(Cw(neighbors));
    neighborColors = setdiff(neighborColors, V0c);
    if isempty(neighborColors)
        disp('V0 has no different colored neighbors');
        P(s) = P(s-1);
        continue;
    end
    
    % step 3: choose a new color & label for V0
    c = randsample([neighborColors max(Cw)+1], 1);
    if c == max(Cw)+1
        newL = randsample([1 2], 1);
    else
        newL = LABELS(find(Cw == c, 1));
    end
    
    LABELSp = LABELS;
    LABELSp(V0) = newL;
    
    
    % step 5: compute the acceptance ratio (log likelihood)
    Pp = pottsPost(G0, LABELSp, T);    
    a = exp( Pp - P(s-1));  % <- no proposal densities for now to keep it simple
    
    % step 6: accept or reject (W or Wp)
    r = rand(1);
    if r <= a
        % display the V0 and the region we've chosen to merge it to
        figure(1234); clf;
        gplotl(W,centers,LABELS,Iraw);
        gplotregion(W,centers, Cw, c, [0 .6 0], 's-');
        gplotregion(V0a,centers, Cw, V0c, [0 1 0], 'o-');
        pause(0.06); refresh;
        
        Cw(V0) = c;     % apply new color c to V0
        % step 4: update Wp to reflect CP's new label, including edges
        W = swc_AdjFromColor(G0, Cw, W, [V0c c neighborColors]);
        %W = swc_AdjFromColor(G0, Cw);
        LABELS = LABELSp;
        P(s) = Pp;
        disp(['accepted sample ' num2str(s) ', a=' num2str(a)]);
    else
        % display the V0 and the region we've chosen to merge it to
        figure(1234); clf;
        gplotl(W,centers,LABELS,Iraw);
        gplotregion(W,centers, Cw, c, [.6 0 0], 's-');
        gplotregion(V0a,centers, Cw, V0c, [1 0 0], 'o-');
        pause(0.06); refresh;
        
        P(s) = P(s-1);
        disp(['rejected sample ' num2str(s) ', a=' num2str(a)]);
    end
    
    figure(445); clf;
    plot(P); grid on;  axis([1 floor(s/100)*100 + 100 min(P(P~=0))  floor(max(P)/100)*100 + 100]);
        
    
   %keyboard;
   
end









% % plot the labeled graph
% figure; imshow(Iraw); axis image off; set(gca, 'Position', [0 0 1 1]); hold on; 
% colors = bone(max(LABELS(:)));
% for l = 1:max(LABELS(:))
%     A = sparse([],[],[], size(W,1), size(W,1),0);
%     members = find(LABELS == l)';
%     for m = members
%        A(m,:) = W(m,:); 
%     end
%     A = max(A,A');
%     %hold on; plot(centers(members,2), centers(members,1), '.', 'Color', colors(c,:));
%     gplot2(A, [centers(:,2) centers(:,1)], '.-', 'Color', colors(l,:));
% end



% % plot the adjacency graph SLOWER KEVIN METHOD
% figure; imshow(Ig);  axis image off; set(gca, 'Position', [0 0 1 1]);
% hold on; plot(centers(:,2), centers(:,1), 'b.');  %switch x and y for plotting
% for l = 1:max(L(:))
%    adj = find(G0(l,:)); 
%    if ~isempty(adj)
%        for a = adj
%             line([centers(l,2) centers(a,2)], [centers(l,1) centers(a,1)]); % switch x and y for plotting
%        end
%    end
% end
% 



% % label the partitioned graph
% [numC,C] = graphconncomp(CP, 'directed', 0);
% 
% 
% % plot the colored graph
% figure; imshow(Iraw); axis image off; set(gca, 'Position', [0 0 1 1]); hold on; 
% colors = jet(numC);
% colors = colors(randperm(numC),:);
% for c = 1:numC
%     A = sparse([],[],[], size(CP,1), size(CP,1),0);
%     members = find(C == c)';
%     for m = members
%        A(m,:) = CP(m,:); 
%     end
%     A = max(A,A');
%     %hold on; plot(centers(members,2), centers(members,1), '.', 'Color', colors(c,:));
%     gplot2(A, [centers(:,2) centers(:,1)], '.-', 'Color', colors(c,:));
% end
% 
% % form connected components CP by randomly turning off edges
% 
% 
% % select a connected component V0
% 
% 
% % choose a new label for V0
% 
% 
% % form the new partitioned graph
% 
% 
% % compute acceptance ratio