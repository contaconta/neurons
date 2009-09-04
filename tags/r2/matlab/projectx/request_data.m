function [SET, DATASETS] = request_data(SET, query, N, DATASETS, LEARNERS, c, varargin)
%
% by default, generates new data. set 'update' to update and old set.
%
% DATASETS.prev_queries keeps a list of which queries have been made.
% Previously requested queries will load a truncated list from a file, so
% repeated selection of the same sample points is avoided. Clearing the
% DATASETS.prev_queries field will reset the data selection.
% 
% DATASETS = rmfield(DATASETS, 'prev_queries');
%

% process the optional arguments
[UPDATE, DISPLAY, CASCADE, GENERATE, DATASETS] = process_arguments(query, DATASETS, varargin);

% Sample data points are stored in files to minimize memory cost. Newly
% generated full lists are stored in fileNameNew, while the current working
% list with used samples removed is kept in fileNameCurrent
fileNameNew = [pwd '/temp/FULL_' [DATASETS.LABELME_FOLDERS{:}] '_' query '_posXY' num2str(DATASETS.posXY) '_negXY' num2str(DATASETS.negXY) '_IM' regexprep(num2str(DATASETS.IMSIZE), '\s*', 'x') '.mat'];
fileNameCurrent = [pwd '/temp/CURR_' [DATASETS.LABELME_FOLDERS{:}] '_' query '_posXY' num2str(DATASETS.posXY) '_negXY' num2str(DATASETS.negXY) '_IM' regexprep(num2str(DATASETS.IMSIZE), '\s*', 'x') '.mat'];

% generate or load from file a shuffled list of sample locations L matching
% the query. if updating, use the current list with used samples removed,
% if starting a new experiment, load all of the possible samples in list

if GENERATE
    if exist(fileNameNew, 'file')
        load(fileNameNew);
        disp(['... loaded ' fileNameNew]);
    else
        L = generate_sample_list(SET,DATASETS,query, DISPLAY);
        save(fileNameNew, 'L');
    end
    disp(['... ' num2str(size(L,1)) ' examples exist for: "' query '".']);
else      
    load(fileNameCurrent);
    disp(['... loaded ' fileNameCurrent]);
end

% extract N examples from L, and place them in SET.
[SET, L] = p_extract_patches(L, N, SET, LEARNERS, UPDATE, DATASETS, c, CASCADE, DISPLAY, query); %#ok<NASGU>

% save the shortened list of samples so we can draw more in the future
save(fileNameCurrent, 'L');





% =========================================================================
% TODO: describe
%
%
function  [UPDATE, DISPLAY, CASCADE, GENERATE, DATASETS] = process_arguments(query, DATASETS, v)

UPDATE = 0; DISPLAY = 0; CASCADE = []; GENERATE = 1;

for i = 1:length(v)
    if isstruct(v{i})
        CASCADE = v{i};
    else
        if strcmp(v{i}, 'update')
            UPDATE = 1;
        end
        if strcmp(v{i}, 'display')
            DISPLAY = 1;
        end
        if strcmp(v{i}, 'generate')
            GENERATE = 1;
        end
    end
end

if isfield(DATASETS, 'prev_queries')
    if ~isempty(DATASETS.prev_queries)
        if find(strcmp(query, DATASETS.prev_queries))
            GENERATE = 0;
            DATASETS.prev_queries = unique( {DATASETS.prev_queries{:}, query}); %#ok<CCAT>
        end
    end
end

if GENERATE
    if isfield(DATASETS, 'prev_queries')
        DATASETS.prev_queries = unique( {DATASETS.prev_queries{:}, query}); %#ok<CCAT>
    else
        DATASETS.prev_queries = {query};
    end
end


% =========================================================================
% generate a list of sample data points matching the query
% from the images in SET and the parameters of DATASETs, we will generate a
% shuffled list of points, L, that contains matches to the labelme QUERY.
%
%
%
function  L = generate_sample_list(SET,DATASETS,query, DISPLAY)

% TODO: ONLY LOAD / WORK ON THE PROPER LIST!
BUFR = floor(max(DATASETS.IMSIZE/2)) + 1;
BUFC = floor(max(DATASETS.IMSIZE/2)) + 1;
BUF = max(BUFR,BUFC);
L = [];


% TODO: separate the data set into training and validation sources
% if length(SET.SourceImages) > 1
%     TLIST = 1:floor(length(SET.SourceImages)/2);
%     VLIST = floor(length(SET.SourceImages)/2)+1:length(SET.SourceImages);
% else
%     error('Not enough training images!');
% end

% split the query string into components    
qsplit = regexp(query, '\s*,\s*', 'split');


% LOOP through the SourceImages
for i = 1:length(SET.SourceImages)

    % retrieve polygons corresponding to each query term in this image
    xo={};yo={};xi={};yi={};itype={};
    for q = 1:length(qsplit)
        Q = qsplit{q};
        if strcmp(Q(1), '!')
            Q = regexprep(regexprep(Q, '!', ''), '&', ',');
            [xo{length(xo)+1},yo{length(yo)+1}] = LMobjectpolygon(DATASETS.LabelMeIndex(i).annotation, Q); %#ok<AGROW>
        else
            if strcmp(Q(1), '%')
                itype{length(yi)+1} = '%'; %#ok<AGROW>
            elseif strcmp(Q(1), '@')
                itype{length(yi)+1} = '@';%#ok<AGROW>
            else
                itype{length(yi)+1} = 'whole';%#ok<AGROW>
            end
            Q = regexprep(regexprep(Q, '%', ''), '@', '');
            [xi{length(xi)+1},yi{length(yi)+1}] = LMobjectpolygon(DATASETS.LabelMeIndex(i).annotation, Q); %#ok<AGROW>
        end
    end

    % locs will store sampled query match locations for image i
    locs = [];

    % polygons from ! querys which need exterior points
    if ~isempty(xo) && ~isempty(xo{1})
        xout = [xo{:}];
        yout = [yo{:}];  
        %disp([ num2str(length(yout)) ' exterior (!) polygons found in image ' DATASETS.LabelMeIndex(i).annotation.filename]);
        [onode ocnect] = multipoly(xout,yout);


        %[u,v] = meshgrid(BUF:DATASETS.negXY:size(SET.SourceImages{1},2)-BUF-1,BUF:DATASETS.negXY:size(SET.SourceImages{1},1)-BUF-1);
        [u,v] = meshgrid(BUFR+1:DATASETS.posXY:size(SET.SourceImages{i},2)-BUFR+1,BUFC+1:DATASETS.posXY:size(SET.SourceImages{i},1)-BUFC+1);                 
        %[u,v] = meshgrid(1:DATASETS.posXY:size(SET.SourceImages{1},2)-DATASETS.IMSIZE(1)+1,1:DATASETS.posXY:size(SET.SourceImages{1},1)-DATASETS.IMSIZE(2)+1);            
        p = [u(:),v(:)];
        [in,bnd]=inpoly(p,onode,ocnect); %#ok<NASGU>
        p = p - repmat([BUFR BUFC],size(u(:)));  
        nin = find(~in);
        locs = [locs; repmat(i,size(nin)), p(nin,1), p(nin,2), repmat(DATASETS.IMSIZE,size(nin))]; %#ok<AGROW>
    end

    % polygons from regular queries with need interior points, boundary points, or whole examples
    if ~isempty(xi)
        %disp([ num2str(numel([xi{:}])) ' interior polygons found in image ' DATASETS.LabelMeIndex(i).annotation.filename]);
        for l=1:length(xi)
            if ~isempty(xi{l})
                switch itype{l}
                    % the entire object is requested
                    case 'whole'
                        x = xi{l};
                        y = yi{l};
                        xmax = cellfun(@max, x)';
                        xmin = cellfun(@min, x)';
                        ymax = cellfun(@max, y)';
                        ymin = cellfun(@min, y)';
                        locs = [locs; repmat(i, size(xmax)), xmin, ymin, xmax-xmin+1, ymax-ymin+1]; %#ok<AGROW>

                    % sample points from boundary pixels are requested
                    case '%'
                        [inode icnect] = multipoly(xi{l},yi{l});
                        [u,v] = meshgrid(BUFR+1:DATASETS.posXY:size(SET.SourceImages{i},2)-BUFR+1,BUFC+1:DATASETS.posXY:size(SET.SourceImages{i},1)-BUFC+1);
                        %[u,v] = meshgrid(1:DATASETS.posXY:size(SET.SourceImages{1},2)-DATASETS.IMSIZE(1)+1,1:DATASETS.posXY:size(SET.SourceImages{1},1)-DATASETS.IMSIZE(2)+1);
                        p = [u(:),v(:)] - repmat([BUFR BUFC],size(u(:)));                        
                        [in,bnd]=inpoly(p,inode,icnect);
                        nin = find(bnd);
                        locs = [locs; repmat(i,size(nin)), p(bnd,1), p(bnd,2), repmat(DATASETS.IMSIZE,size(nin))]; %#ok<AGROW>

                    % sample points from interior of object requested
                    case '@'
                        [inode icnect] = multipoly(xi{l},yi{l});
                        [u,v] = meshgrid(BUFR+1:DATASETS.posXY:size(SET.SourceImages{i},2)-BUFR+1,BUFC+1:DATASETS.posXY:size(SET.SourceImages{i},1)-BUFC+1);
                        %[u,v] = meshgrid(1:DATASETS.posXY:size(SET.SourceImages{1},2)-DATASETS.IMSIZE(1)+1,1:DATASETS.posXY:size(SET.SourceImages{1},1)-DATASETS.IMSIZE(2)+1);
                        p = [u(:),v(:)];  %  - repmat([BUFR BUFC],size(u(:)));                            
                        [in,bnd]=inpoly(p,inode,icnect); %#ok<NASGU>
                        p = p - repmat([BUFR BUFC],size(u(:)));      
                        nin = find(in);
                        locs = [locs; repmat(i,size(nin)), p(in,1), p(in,2), repmat(DATASETS.IMSIZE,size(nin))]; %#ok<AGROW>
                end
            end
        end        
    end

    % ====================== TEMP FOR DISPLAY ONLY! =================== 
    if DISPLAY
        imshow(SET.SourceImages{i}); hold on; 
        if ~isempty(locs)
            % plot the point samples
            pts = locs(:,4) == DATASETS.IMSIZE(1);
            plot(locs(pts,2)+BUF+1, locs(pts,3)+BUF+1,'r.'); hold off;
            % plot the whole samples
            patches = find(locs(:,4)~=DATASETS.IMSIZE(1));
            for j = 1:length(patches)
                lx = [locs(patches(j),2) locs(patches(j),2)+locs(patches(j),4)-1 locs(patches(j),2)+locs(patches(j),4)-1 locs(patches(j),2) locs(patches(j),2)]; 
                ly = [locs(patches(j),3) locs(patches(j),3) locs(patches(j),3)+locs(patches(j),5)-1 locs(patches(j),3)+locs(patches(j),5)-1 locs(patches(j),3)]; 

                line(lx,ly, 'Color', [1 0 0 ]);
            end
        end
        input('Press Enter to contine');
    end
    % =================================================================

    % append locs to the master list, L
    L = [L; locs]; %#ok<AGROW>
    
    %S = sprintf('%8d samples extracted from %3d polygons in image %s', size(locs,1), numel([xi{:}]) + numel([xo{:}]), DATASETS.LabelMeIndex(i).annotation.filename);
    %disp(S);
end

% shuffle the locations, so we can quickly sample from the top
L = L(randperm(size(L,1)),:);


