% set necessary paths
addpath([pwd '/utils/']);
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

% plot the original image
figure; imshow(Iraw);  axis image off; set(gca, 'Position', [0 0 1 1]);

% plot the segmentation with average gray levels
figure; imshow(Ig);  axis image off; set(gca, 'Position', [0 0 1 1]);

% plot the superpixel centers
figure; imshow(Ig);  axis image off; set(gca, 'Position', [0 0 1 1]);
hold on; plot(centers(:,2), centers(:,1), 'b.');

% plot the adjacency graph
figure; imshow(Ig);  axis image off; set(gca, 'Position', [0 0 1 1]);
hold on; plot(centers(:,2), centers(:,1), 'b.');  %switch x and y for plotting
for l = 1:max(L(:))
   adj = find(G0(l,:)); 
   if ~isempty(adj)
       for a = adj
            line([centers(l,2) centers(a,2)], [centers(l,1) centers(a,1)]); % switch x and y for plotting
       end
   end
end


% partition the graph

% label the partitioned graph

% form connected components CP by randomly turning off edges


% select a connected component V0


% choose a new label for V0


% form the new partitioned graph


% compute acceptance ratio