% ========= parameters ==================
S = 20000;  % number of samples
tic;

% Labels :
% 1 = Background
% 2 = Boundary
% 3 = Mitochondria interior
%LabelList = [1 2 3];
LabelList = [1 2];

% Debug parameters
exportToFileOnly = false;
createMovie = true;
report = fopen('report.txt','w');
displayOn = true;
if createMovie
  displayOn = true;
end
if exportToFileOnly
  h = figure('Visible', 'off');
  createMovie = false;
  displayOn = false;
end

% set necessary paths
addpath([pwd '/utils/']);
addpath([pwd '/bin/']);
addpath([pwd '/utils/libsvm-mat-2.89-3/']);
huttPath = '/osshare/Work/software/huttenlocher_segment/';
imPath = [pwd '/images/'];

% fix the random number stream
st = RandStream.create('mt19937ar','seed',5489);  RandStream.setDefaultStream(st);  %rand('twister', 100);    % seed for Matlab 7.8 (?)

% load an image we want to play with
%Iraw = imread([imPath 'test.png']);
%Iraw = imread('/osshare/Work/neurons/matlab/swc/temp/seg_plus_labels/FIBSLICE0100.png');
%Iraw = imread('/osshare/Work/Data/LabelMe/Images/fibsem/FIBSLICE0002.png');
%Iraw = imread('/localhome/aurelien/usr/share/Data/LabelMe/Images/FIBSLICE/FIBSLICE0002.png');
%Iraw = imread('/home/alboot/usr/share/Data/LabelMe/Images/FIBSLICE/FIBSLICE0002.png');
Iraw = imread('images/FIBSLICE0002.png');

useGroundTruth = true;
rescaleData = true;
kernelType = 2; % RBF kernel
if useGroundTruth
  IGroundTruth = imread('images/annotation0002.png');
  IGroundTruth = IGroundTruth(:,:,1);
  IGroundTruth = IGroundTruth(1:480,1:640);
else
  IGroundTruth = 0;
end

% load superpixels or atomic regions as a label matrix, L
% disp('Loading the superpixel segmentation image.');
% HUT = imread([imPath 'testHUTT.ppm']);
% disp('Assigning labels to each superpixel in the segmentation image (slow).');
% L = rgb2label(HUT);
%L = readRKLabel('/osshare/Work/neurons/matlab/swc/temp/seg_plus_labels/FIBSLICE0100.dat', [480 640 ])';
% L = readRKLabel('/osshare/Work/neurons/matlab/swc/temp/seg_plus_labels/FIBSLICE0002.dat', [1536 2048 ])'; 
%L = readRKLabel('/localhome/aurelien/Sources/EM/Superpixels/seg_plus_labels_corrected/FIBSLICE0400.dat', [1536 2048 ])';
L = readRKLabel('temp/labels/FIBSLICE0002.dat', [1536 2048])';

Iraw = Iraw(1:480, 1:640);
L = L(1:480,1:640);

% Rename labels
Llist = unique(L);
for l = 1:length(Llist)
    L(L == Llist(l)) = l;
end

% 
% %keyboard;
% 
% % G contains average gray levels of I for regions in L 
% disp('Filling greylevels in superpixel segmentation image.');
% L = medfilt2(L);
% for l = 1:length(Llist)
%     L(L == Llist(l)) = l;
% end
% Ig = label2gray(L,Iraw); Ig = uint8(round(Ig));
% 
% disp('cleaning up RKs mess');
% missing = (Ig == 0);
% BW = bwlabel(missing);
% for l=1:max(BW(:))
%     if length(find(BW == l)) ==1
%         L(BW==l) = unique(L(find(BW ==l) -1));
%     else
%         L(BW==l) = max(max(L)) + 1;
%     end
% end
    
%load LRK.mat;
disp('Filling greylevels in superpixel segmentation image.');
Ig = label2gray(L,Iraw); Ig = uint8(round(Ig));

% extract and adjacency matrix and list from L
if ~exist('G0','var')
  disp('Extracting adjacency graph G0 from superpixel segmentation image.');
  [G0, G0list] = adjacency(L);
else
    disp('Adjacency graph G0 was already computed.');
end

% create a list of superpixel center locations and pixel lists
disp('Computing superpixel center locations.');
centers = zeros(max(L(:)),1); pixelList = cell([1 size(G0,1)]);
for l = 1:max(L(:))
    pixelList{l} = find(L == l); 
    [r,c] = ind2sub(size(L), pixelList{l});
    centers(l,1) = mean(r);
    centers(l,2) = mean(c);
end
% plot(centers(:,1),centers(:,2),'*')

% precompute the KL divergences
disp('Precomputing the KL divergences.');
KL = edgeKL(Iraw, pixelList, G0, 1);

% initialize the SVM model
if useGroundTruth==false
  if ~exist('model', 'var')
    disp('Computing the SVM model.');
    feature_vectors = [pwd '/temp/Model-0-4200-3-sup/feature_vectors'];
    [label_vector, instance_matrix] = libsvmread(feature_vectors);
    training_label = label_vector(1:4000,:);
    training_instance = instance_matrix(1:4000,:);

    [model,minI,maxI] = loadModel(training_label,training_instance, ...
                                  rescaleData,kernelType);
  end
else
  disp('No model computed. Ground truth data will be used.');
  model = 0;
  minI = 0;
  maxI = 0;
end

%keyboard;

%create an initial partition of the graph
disp('Creating a random initial partiton W of the graph.');
B1 = .28;  B1 = 1;  B1 = .5;
W = swc_CP1(G0, B1, KL);         % make initial cuts

% Find the strongly connected components (clique)
% Cw is a vector indicating to which component each node belongs
[numCw,Cw] = graphconncomp(W, 'directed', 0);   % color the graph
%[numCw,Cw] = graphconncomp(W);   % color the graph
%keyboard;
W = swc_AdjFromColor(G0,Cw);                    % fill in missing adjacency edges

% assign labels to the initial partition
disp('Randomly assigning labels to each region in the graph.')
LABELS = zeros(size(Cw));
for c = 1:numCw
    members = find(Cw == c)';
    
    % FIXME : The following doesn't work because we pass a set of
    % pixels belonging to a region butthe SVM was trained
    % using a superpixel and its immediate neighbors
    if useGroundTruth
      lpixels = cell2mat(pixelList(members)');
      LABELS(members) = getMostFrequentLabel(lpixels,IGroundTruth);
      %i2 = zeros(size(IGroundTruth),'uint8');
      %i2(lpixels) = Iraw(lpixels);
      %imshow(i2);
      %keyboard
    else
      pixels = Iraw(cell2mat(pixelList(members)'));
      [predicted_label, accuracy, pb] = getLikelihood(pixels, model,minI,maxI,rescaleData);    
      LABELS(members) = find(pb == max(pb),1);
    end

    %LABELS(members) = 1;
    

    %LABELS(members) = randsample(LabelList,1);
%     if rand(1) < .5
%         LABELS(members) = 1;
%     else
%         LABELS(members) = 2;
%     end
end

% % plot the initial partition
% figure; gplotl(W,centers,LABELS,Iraw); figure; gplotc(W, centers, Cw, Iraw);

% set the annealing schedule
T = .4055; T = 1.5; 
Bstart = 10;  Bend = .2;
%B = [Bstart linterp([1 S], [Bstart Bend], 2:S-1) Bend];
B = Bstart*ones([1 S]);

% compute the initial posterior (log posterior)
disp('Computing initial posterior');
P = zeros([1 S]);
%P(1) = pottsPost(G0, Cw, T);
%P(1) = swc_posterior(W, LABELS, model, minI, maxI, B(1), pixelList,Iraw);
Plist = swc_post(W, LABELS, model, minI, maxI, rescaleData, IGroundTruth, ...
                 useGroundTruth, pixelList, Iraw, [], 'init');
P(1) = sum(Plist);

%% ===================== Metropolis-Hastings ============================
disp('Applying metropolis-hastings with Swendson-Wang cuts.')

%keyboard;
%% build the markov chain
for s = 2:S
    
    % step 1: select a seed vertex v
    v = randsample(size(W,1),1);
    
    % step 2: grow a region V0 with color V0c
    B_CUT = .6;
    [V0, V0a] = swc_swc2_1(W, B_CUT, v, KL);  	% get a list of the members of V0
    V0c = Cw(V0(1));                    % the current color of V0
    disp(['V0 size=' num2str(size(V0,2)) ', NbAdj=' num2str(sum(sum(V0a==1)))])

    % determine what are the neighbors colors
    [neighbors,junk] = find(G0(:,V0));
    neighbors = unique(setdiff(neighbors, V0));
    
    neighborColors = unique(Cw(neighbors));
    neighborColors = setdiff(neighborColors, V0c);
%     if isempty(neighborColors)
%         P(s) = P(s-1); %disp('V0 has no different colored neighbors');
%         continue;
%     end
    
    % step 3: choose a new color & label for V0
    newColor = ones(1,max(1,size(neighborColors,2))).*(max(Cw)+1);
    c = randsample([neighborColors newColor], 1);
    %c = randsample([neighborColors (max(Cw)+1)], 1);
    if c == max(Cw)+1
        newL = randsample(LabelList, 1);
        type_move = 'Split';
    else
        newL = LABELS(find(Cw == c, 1));
        type_move = 'Merge';
    end
    %newL = randsample(LabelList, 1);

    LABELSp = LABELS;
    LABELSp(V0) = newL;
    
    
    % step 5: compute the acceptance ratio (log likelihood)
    Cwp = Cw;
    Cwp(V0) = c;
    %Pp = pottsPost(G0, Cwp, T);   
    %Pp = swc_posterior(W, LABELSp, model, minI, maxI, B(s), pixelList,Iraw);
    Pplist = swc_post(W, LABELSp, model, minI, maxI, rescaleData, IGroundTruth, ...
                      useGroundTruth, pixelList, Iraw, V0, Plist);
    
    % Compute cardinality
    Pp = sum(Pplist(V0));
    Pold = sum(Plist(V0));
    
    a = exp( B(s)* (Pp - Pold));
%     a = exp( Pp - P(s-1));  % <- no proposal densities for now to keep it simple
    
    % step 6: accept or reject (W or Wp)
    r = rand(1);
    if r <= a
         % display the V0 and the region we've chosen to merge it to
         if displayOn
           figure(1234); clf;
           gplotl(W,centers,LABELS,Iraw);
           gplotregion(W,centers, Cw, c, [0 .6 0], 's-');
           gplotregion(V0a,centers, Cw, V0c, [0 1 0], 'o-');
           pause(0.06); refresh;
         end
%         figure(1234); clf;
%         Itemp = Iraw;
%         for v = V0
%             Itemp(pixelList{v}) = 255;
%         end
%         gplotl(W,centers,LABELS,Itemp);

        if createMovie
          txtReport = ['Accepted ' num2str(s) '(a=' num2str(a) ...
                       '), newL=' num2str(newL) ', ' type_move];
          txtReport = [txtReport ', V0 size=' num2str(size(V0,2)) ', NbAdj=' num2str(sum(sum(V0a==1)))];
          fprintf(report,[txtReport '\n']);
          legend(txtReport);
          eval(['print -f1234 -dpng imgs/img' num2str(s) '.png']);
        end

        
        % step 4: update Wp to reflect CP's new label, including edges
        Cw(V0) = c;     % apply new color c to V0
        W = swc_AdjFromColor(G0, Cw, W, [V0c c neighborColors]);
        [numCw,Cw] = graphconncomp(W, 'directed', 0);
        LABELS = LABELSp;
        %P(s) = Pp;
        P(s) = sum(Pplist);
         disp(['accepted sample ' num2str(s) ', a=' num2str(a) ', newL=' num2str(newL)]);
         
%          if displayOn
%            figure(1235); clf;
%            gplotl(W,centers,LABELS,Iraw);
%            gplotregion(W,centers, Cw, c, [0 .6 0], 's-');
%            gplotregion(V0a,centers, Cw, V0c, [0 1 0], 'o-');
%            pause(0.06); refresh;
%          end
    else
%         % display the V0 and the region we've chosen to merge it to
         if displayOn
           figure(1234); clf;
           gplotl(W,centers,LABELS,Iraw);
           gplotregion(W,centers, Cw, c, [.6 0 0], 's-');
           gplotregion(V0a,centers, Cw, V0c, [1 0 0], 'o-');
           pause(0.06); refresh;
         end
        
         if createMovie
          txtReport = ['Rejected ' num2str(s) '(a=' num2str(a) ...
                       '), newL=' num2str(newL) ', ' type_move];           
           txtReport = [txtReport ', V0 size=' num2str(size(V0,2)) ', NbAdj=' num2str(sum(sum(V0a==1)))];
           fprintf(report,[txtReport '\n']);           
           legend(txtReport);
           eval(['print -f1234 -dpng imgs/img' num2str(s) '.png']);
         end
         
         P(s) = P(s-1);
         disp(['rejected sample ' num2str(s) ', a=' num2str(a) ', L=' ...
               num2str(newL)]);
         
%          if displayOn
%            figure(1235); clf;
%            gplotl(W,centers,LABELS,Iraw);
%            gplotregion(W,centers, Cw, c, [0 .6 0], 's-');
%            gplotregion(V0a,centers, Cw, V0c, [0 1 0], 'o-');
%            pause(0.06); refresh;
%          end
    end
    
    if createMovie && mod(s,20) == 0
      % Save file
      fclose(report);
      report = fopen('report.txt','a');
    end
    
    if ~displayOn && mod(s,100) == 0
      if exportToFileOnly
        clf;
        gplotl(W,centers,LABELS,Iraw);
        eval(['print -f' int2str(h) ' -dpng imgs/img' num2str(s) '.png']);
      else      
        figure(1234); clf;
        gplotl(W,centers,LABELS,Iraw);
        %gplotc(W,centers,LABELS,Iraw);
        %pause(0.06); refresh;
        figure(445); clf;
        plot(P); grid on;  axis([1 floor(s/100)*100 + 100 min(P(P~=0))  floor(max(P)/100)*100 + 100]);
        pause(0.06); refresh;
        disp([' sample ' num2str(s) ' NUMCOLORS=' num2str(max(Cw)) ', P='  num2str(P(s)) ', a=' num2str(a)]);
      end
    end
    
    % plot the progress of the posterior estimate
%     figure(445); clf;
%     plot(P); grid on;  axis([1 floor(s/100)*100 + 100 min(P(P~=0))  floor(max(P)/100)*100 + 100]);
        
    
    keyboard;
    %keydown = waitforbuttonpress;
   
end



toc





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


