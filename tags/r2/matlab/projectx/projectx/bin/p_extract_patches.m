function [SET, L] = p_extract_patches(L, N, SET, LEARNERS, UPDATE, DATASETS, c, CASCADE, DISPLAY, query)
% p_extract_patches(L, N, SET, LEARNERS)
% p_extract_patches(L, N, SET, LEARNERS, 0)
% p_extract_patches(L, N, SET, LEARNERS, 1, CASCADE)
%
%  N = # samples needed, c = class label to update, L = list of sample
%  coordinates, DISPLAY = [0,1] shows intermediate steps

% ======== HANDLE SPECIAL CASE LEARNERS HERE! ====================
if strmatch('HA_', LEARNERS.types)
    LE(1) = 1;
end

%% Fill the initial SET with positive and negative examples
% no need to check if false alarms are generated!
if ~UPDATE
    % take the first N sample locations from L as examples, make list of corresponding images
    if N > size(L,1); error(['Error extract_patches: ' num2str(N) ' samples were requested when only ' num2str(size(L,1)) ' exist!']); end;
    
    % l is a structure containing N processed examples and their
    % coordinates, as well as special data (e.g. integral images)
    l = collect_process_samples(L,N,SET,DATASETS,DISPLAY,LE);
    

    % place the images, coordinates, and class labels in SET
    if ~isfield(SET, 'Images')
        SET.Images = {};  SET.Coords = []; SET.class = [];
    end
    
    % ======== HANDLE SPECIAL CASE LEARNERS HERE! =================
    if LE(1); SET.IntImages(length(SET.Images)+1:length(SET.Images)+length(l.images)) = deal(l.intimages(:)); SET.IntImages = SET.IntImages(:); end;
    
    SET.Images(length(SET.Images)+1:length(SET.Images)+length(l.images)) = deal(l.images(:));
    SET.Images = SET.Images(:);
    SET.Coords(size(SET.Coords,1)+1:size(SET.Coords,1)+size(l.coords,1), :) = l.coords;
    SET.class(length(SET.class)+1:length(SET.class)+length(l.images)) = c;
    SET.class = SET.class(:);
    SET = orderfields(SET);
        
    % remove the coords we've used from the list
    L = L(N:length(L),:);

else 
    %% Replace true negative samples with new samples providing FP's
    
    % parameters
    CHUNK = 1000;           % define how many examples we should handle at a time
    usedExamples = 0;       % keep track of how many examples from L we've searched through
    
    % make sure we were passed a CASCADE to classify with
    if isempty(CASCADE); error('p_extract_patches: no CASCADE passed to update the data set!'); end;
        
   	% find the TN's that we must replace with FP's
    gt = [SET(:).class]';  
    C = p_classify_cascade(CASCADE, SET);
    %C = randsample([-1 1], length(gt), true);  % DEBUG: randomly sampling the classifier response for debugging
    TNs = rocstats(C, gt,'TNlist'); 
    
    % TNs contains indexes of SET examples which need to be
    % replaced, N is the number of FP's we need to generate
    N = length(TNs);
    
    % as we collect False Positives, we will store them in a data set
    % structure FP which looks like SET so the classifier can handle it
    FP.Images = {}; FP.Coords = []; FP.precomputed = 0;
    
    while length(FP.Images) < N
      	% select CHUNK examples
        if CHUNK > size(L,1); error('p_extract_patches: WE HAVE RUN OUT OF NEGATIVE TRAINING SAMPLES!'); end;
        l = collect_process_samples(L,CHUNK,SET,DATASETS,DISPLAY,LE);
        
        % place examples in FP
        % ======== HANDLE SPECIAL CASE LEARNERS HERE! =================
        if LE(1); FP.IntImages(length(FP.Images)+1:length(FP.Images)+length(l.images)) = deal(l.intimages(:)); FP.IntImages = FP.IntImages(:); end;

        FP.Images(length(FP.Images)+1:length(FP.Images)+length(l.images)) = deal(l.images(:));
        FP.Images = FP.Images(:);
        FP.Coords(size(FP.Coords,1)+1:size(FP.Coords,1)+size(l.coords,1), :) = l.coords;
    
        % classify FP
        gt = repmat(-1, size(FP.Images));  
        C = p_classify_cascade(CASCADE, FP);
        %C = randsample([-1 1], length(gt), true);  % DEBUG: randomly sampling the classifier response for debugging
    
        % remove all non-False Positives from FP
        FPlist = rocstats(C, gt,'FPlist'); %disp(['FPlist length = ' num2str(length(FPlist)) ]);
        FP.Images = FP.Images(FPlist);
        FP.Coords = FP.Coords(FPlist,:);
        % ======== HANDLE SPECIAL CASE LEARNERS HERE! =================
        if LE(1); FP.IntImages = FP.IntImages(FPlist); end;

        disp(['   ....searching through ' num2str(CHUNK) ' examples for False Positives.  Found (' num2str(length(FP.Images)) '/' num2str(N) ') ']);
        
        % keep track of how many examples from L we've searched through
        usedExamples = usedExamples + CHUNK;
    end
   
    % replace TNs in SET with N recently collected FP examples
    SET.Images(TNs) = FP.Images(1:N);
    SET.Coords(TNs,:) = FP.Coords(1:N,:);
    % ======== HANDLE SPECIAL CASE LEARNERS HERE! =================
    if LE(1); SET.IntImages(TNs) = FP.IntImages(1:N); end;
   
    
    
   	% remove the examples from L we've searched through from the list
   	L = L(usedExamples:length(L),:);
    
    
end


disp(['... Selected ' num2str(N) ' examples. ' num2str(size(L,1)) ' examples remain for: "' query '".']);








%% ==== collect_process_samples ===========================================
%
%
%
%
%
%
function lists = collect_process_samples(L,N,SET,DATASETS,DISPLAY,LE)

coords = L(1:N,:);
coords = sortrows(coords);
imlist = unique(coords(:,1))';

images = cell(size(coords(:,1)));
% ======== HANDLE SPECIAL CASE LEARNERS HERE! ====================
if LE(1); intimages = cell(size(images)); end;

% perform preprocessing for the large source images so we don't repeat
% effort when performing on patches NOTE: DEPENDING ON HOW MANY SAMPLES WE
% TAKE, IT MIGHT BE QUICKER TO COMPUTE SOME FEATURES ON PATCHES, NOT THE
% SOURCE IMAGE. Some features such as Rays will suffer for this, though.
for i = imlist

    I = SET.SourceImages{i};

    % for the moment, we want to set Images to grayscale
    if length(size(I)) > 2;  I = rgb2gray(I); end;

    % ======== HANDLE SPECIAL CASE LEARNERS HERE! ====================
    % if haar learners are used, we need to compute the integral image
    %if LE(1); IntImage = integral_image(I); end;

    if DISPLAY; imshow(I); end; %display for debugging purposes only

    % extract the patching defined by coords belonging to image i
    for j = find(coords(:,1) == i)'

        images{j} = I(coords(j,3):coords(j,3)+coords(j,5)-1,coords(j,2):coords(j,2)+coords(j,4)-1);

        if ~isequal(size(images{j}),DATASETS.IMSIZE)
            disp('needed to resize!');
            keyboard;
            images{j} = imresize(images{j}, DATASETS.IMSIZE);
        end

        % ======== HANDLE SPECIAL CASE LEARNERS HERE! =================
        if LE(1); intimages{j} = integral_image(images{j}); end


        % draw patch boxes on image
        if DISPLAY; line([coords(j,2) coords(j,2)+coords(j,4)-1 coords(j,2)+coords(j,4)-1 coords(j,2) coords(j,2) ], [coords(j,3) coords(j,3) coords(j,3)+coords(j,5)-1 coords(j,3)+coords(j,5)-1 coords(j,3) ], 'Color', [1 0 0]); end;
    end
    if DISPLAY; input('Press ENTER to continue.'); end;
end

% store in a structure for easy passing
lists.coords = coords;
lists.images = images;
if LE(1); lists.intimages = intimages; end

