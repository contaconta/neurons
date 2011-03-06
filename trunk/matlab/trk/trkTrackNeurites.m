function R = trkTrackNeurites(R)

W_THRESH = 300;
MIN_TRACK_LENGTH = 3;


% extract a list of neurites and important properties
disp('...extracting neurites');tic;
[N Nlist R] = getNeurites(R); toc;


% make and adjacency matrix of neurites
disp('...making adjacency matrix');tic;
A = make_adjacency(Nlist,1,N); toc;

% fill out all the distances in the adjacency matrix
edges = find(A == 1);
W = A;
for i = 1:length(edges)
    [r,c] = ind2sub(size(A), edges(i));
    W(r,c) = trkNeuriteDistance(N(r), N(c));
end

%% apply the greedy tracking algorithm to link detections
disp('...greedy tracking'); tic;
T = trkGreedyConnect(W,A,N,W_THRESH); toc;


%% get the track labels from T assigned to each detection
disp('...graph coloring'); tic;
[T ntracks] = trkGraphColoring(T, MIN_TRACK_LENGTH);  toc; clear T; %#ok<ASGLU>

%% assign NeuriteTrack ID's to each neurite
for n = 1:length(N)
    N(n).NeuriteTrack = ntracks(n);
end

R.N = N;





% %% temporary frame-rendering
% imgRootFolder = '/home/ksmith/data/Sinergia/Basel/';
% imgFolder = [imgRootFolder R.GlobalMeasures.Date '/' R.GlobalMeasures.AssayPosition '/green/'];
% Gfiles = dir([imgFolder '*.tif']);
% mv = getImgFiles(imgFolder, Gfiles, R.GlobalMeasures.Length);
% cols1 = jet(6);
% cols1 = cols1(randperm(6),:);
% cols2 = jet(8);
% cols2 = cols2(randperm(8),:);
% cols3 = jet(600);
% cols3 = cols3(randperm(600),:);
% colors = [cols1; cols2; cols3];
% 
% for t = 1:97
%     mv = trkRenderImages3(t,t,R,colors,mv,1);
%     pause;
% end
  



%keyboard;





%% create a structure containing all neurites
function [N Nlist R] = getNeurites(R)

% B = zeros(R.FILAMENTS(1).IMSIZE);

R.FILAMENTS(1).NIdxList = []; % add a new field to the struct

Nlist = cell(1,R.GlobalMeasures.Length);

count = 1;
for d = 1:length(R.D)
    if R.D(d).ID ~= 0
        numNeurites = max(R.FILAMENTS(d).NeuriteID);
        for n = 1:numNeurites
            nIdx = R.FILAMENTS(d).NeuriteID == n;
            [r c] = ind2sub(R.FILAMENTS(d).IMSIZE, R.FILAMENTS(d).PixelIdxList(nIdx));
            nIdx = find(nIdx);
            
            parentsIdx = R.FILAMENTS(d).Parents(nIdx);
            somaIdx = R.FILAMENTS(d).PixelIdxList(parentsIdx(find(R.FILAMENTS(d).NeuriteID(parentsIdx) == 0,1)));
            [somay somax] = ind2sub(R.FILAMENTS(d).IMSIZE, somaIdx);
            
            CC.Connectivity = 8;
            CC.ImageSize = R.FILAMENTS(d).IMSIZE;
            CC.NumObjects = 1;
            CC.PixelIdxList = {R.FILAMENTS(d).PixelIdxList(nIdx)};
            
            Nn = regionprops(CC, 'Centroid', 'Extrema', 'Eccentricity', 'MajorAxisLength', 'MinorAxisLength', 'Orientation');
            Nn.PixelIdxList = R.FILAMENTS(d).PixelIdxList(nIdx);
            
            % vector in direction of the major axis
            dmaj = [cosd(Nn.Orientation) sind(Nn.Orientation)];
            
            % vector pointing towards the nucleus
            dnuc = [R.D(d).Centroid(1) R.D(d).Centroid(2)] - [Nn.Centroid(1) Nn.Centroid(2)];
            Nn.CentroidOffset = [Nn.Centroid(1) Nn.Centroid(2)] - [R.D(d).Centroid(1) R.D(d).Centroid(2)];
            
            % find the dot product of the major axis and nucleus vector
            dnucnorm = dnuc / norm(dnuc);
            Nn.RadialDotProd = abs(dot(dmaj,dnucnorm));
            
            % get some distance measures
            Nn.DistToSomaExtreme = max(sqrt(sum((Nn.Extrema - repmat([somax somay], [size(Nn.Extrema,1),1])).^2,2)));
            dists = sqrt(sum(([c r] - repmat([somax somay], [length(c),1])).^2,2));            
            Nn.DistToSomaMedian = median(dists);
            Nn.DistToSomaMean = mean(dists);
            Nn.DistToSomaStandDev = std(dists);
            Nn.SomaContact = [somax somay] - [R.D(d).Centroid(1) R.D(d).Centroid(2)];
            
            % get filopodia and branching measures
            Nn.FiloCount = length(find(R.FILAMENTS(d).NumKids(nIdx) == 0));
            Nn.FiloMass = length(find(R.FILAMENTS(d).FilopodiaFlag(nIdx) == 1));
            Nn.TotalMass = length(nIdx);
            Nn.BranchCount = length(find(R.FILAMENTS(d).NumKids(nIdx) > 1));
            Nn.FiloPercent = (Nn.FiloMass / Nn.TotalMass) * 100;
            
            % get some essential labels
            Nn.NucleusID = R.D(d).ID;
            neuriteids = R.FILAMENTS(d).NeuriteID(nIdx);
            Nn.NeuriteID = neuriteids(1);
            Nn.Time = R.D(d).Time;
            
            % update a time-ordered list of all the neurite IDs
            Nlist{Nn.Time} = [Nlist{Nn.Time} count];

            % clean up
            Nn = rmfield(Nn, 'Orientation');
            Nn = rmfield(Nn, 'Centroid');
            Nn = rmfield(Nn, 'Extrema');
            Nn = orderfields(Nn);
            
            R.FILAMENTS(d).NIdxList = [R.FILAMENTS(d).NIdxList count];
            
%             Nn
            
            if count == 1
                N = Nn;
            else
                N(count) = Nn;
            end
            count = count + 1;

%             B(Nn.PixelIdxList) = 1;
%             B(somaIdx) = 2;
%             figure(2); imagesc(B); hold on;
%             
%             plot(Nn.Centroid(1), Nn.Centroid(2), 'gx');
%             plot([Nn.Centroid(1) Nn.Centroid(1)+30*dmaj(1)], [Nn.Centroid(2) Nn.Centroid(2)+30*dmaj(2)], 'g-');
%             plot([Nn.Centroid(1) Nn.Centroid(1)+dnuc(1)], [Nn.Centroid(2) Nn.Centroid(2)+dnuc(2)], 'y-');
% 
%             keyboard;
            
        end
    end
end







%% create and adjacency matrix linking nearby detections
function A = make_adjacency(Nlist, WIN_SIZE, N)
Ndetection = length(N);
A = zeros(Ndetection);
for t = 2:length(Nlist)
    for d = 1:length(Nlist{t})
        n_i = Nlist{t}(d);              % neurite index
        nuc_i = N(n_i).NucleusID;       % neurite's nucleus
        filp_i = N(n_i).FiloPercent;
        
        min_t = max(1, t-WIN_SIZE);
        for p = min_t:t-1
            for dp = 1:length(Nlist{p})
                n_p = Nlist{p}(dp);             % past neurite index
                nuc_p = N(n_p).NucleusID;   % past neurite's nucleus
                filp_p = N(n_p).FiloPercent;
                if (nuc_i == nuc_p) && (filp_i < 95) && (filp_p < 95)
                    A(n_i, n_p) = 1;
                end
            end    
        end
    end
end





% function B = drawNeurites(N,nlist,R)
% 
% 
% B = zeros(R.FILAMENTS(1).IMSIZE);
% 
% for i = 1:length(nlist)
%     n = nlist(i);
%     B(N(n).PixelIdxList) = i;
% end
% 
% imagesc(B);
% 
% 
% function mv = getImgFiles(Gfolder, Gfiles, TMAX)
% 
% 
% for t = 1:TMAX
%     G = imread([Gfolder Gfiles(t).name]);
%     
%     if t == 1
%         lims = stretchlim(G);
%     end
%     G8bits = trkTo8Bits(G, lims);
%     
%     % make an output image
%     Ir = mat2gray(G8bits);
%     I(:,:,1) = Ir;
%     I(:,:,2) = Ir;
%     I(:,:,3) = Ir;
%     
%     mv{t} = I;  %#ok<AGROW>
% end
% 
% 
% function J = trkTo8Bits(I, lims)
% 
% 
% %lims = stretchlim(I);         
% J = imadjust(I, lims, []); 
% J = uint8(J/2^8);