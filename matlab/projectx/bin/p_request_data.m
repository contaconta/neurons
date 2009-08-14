function [data, D] = p_request_data(tag_string, N, DATASETS, varargin)
%% P_REQUEST_DATA
%
%   [data, D] = p_request_data('query', N, DATASETS, ...) requests N examples from
%   a LabelMe database selected using the query string. It returns DATA, a 
%   cell containing the N examples as well as D, the LabelMe index
%   containing query results which have not yet been requested.  To request
%   a previously unseen example, pass D as an optional argument.
%       
%   Example:
%   -----------
%   [data, D] = p_request_data('car', 1, DATASETS);  % request 1 car
%   [data, D] = p_request_data('car', 3, DATASETS, D);  % request the next 3 cars
%   [data, D] = p_request_data('non car', 100, DATASETS, D);  % request 3 non car examples
%   [data, D] = p_request_data('non car', 100, D, DATASETS, 'SIZE', [49 49]);  % request 3 non car examples of size [49 49]
%
%   See also LMQUERY

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

SHRINK_BORDER = 10;     % for collecting negative samples: buffer between positive regions and negative sample regions

for k = 1:nargin-3
    switch class(varargin{k})
        case 'struct'
            D = varargin{k};
        case 'char'
            if strcmp(varargin{k}, 'SIZE')
                DATASETS.IMSIZE = varargin{k+1};
            end
    end
end

% set a flag if we are looking for a NON-class
NON = regexp(tag_string, 'non', 'once');

% create a new LabelMe index if it hasn't been passed as an argument
if ~exist('D', 'var')
    %D = LMdatabase(DATASETS.HOMEANNOTATIONS);
    %D = LMquery(LMdatabase(DATASETS.HOMEANNOTATIONS), 'folder', 'fibslice');
    D = LMdatabase(DATASETS.HOMEANNOTATIONS, DATASETS.LABELME_FOLDERS);
end
   
data = cell(1,N);       % image example data will be stored in a cell

% select N negative samples
if NON
    Qresult = D;  j = 1:length(D);
    %[Qresult, j] = LMquery(D, 'object.name', strtrim(regexprep(tag_string, 'non', '')) );
    [data, D] = get_negative_samples(Qresult, N, DATASETS.HOMEIMAGES, DATASETS.IMSIZE, data, D, j, SHRINK_BORDER);   
% select N positive samples
else
%tag_string
    [Qresult, j] = LMquery(D, 'object.name', tag_string);
    if isempty(Qresult); error(['p_request_data: LabelMe cannot find any more examples of type ' tag_string ]);end
    [data, D] = get_positive_samples(Qresult, N, DATASETS.HOMEIMAGES, DATASETS.IMSIZE, data, D, j);    
end








%% ======================= GET_POSITIVE_SAMPLES ===========================
function [data, D] = get_positive_samples(Qresult, N, HOMEIMAGES, IMSIZE, data, D, j)
% pick out N examples from Qresult, and remove them from D so they are not
% re-selected. extract a bounding patch and place it in a cell {data}

i = 1;                  % count the number of query matches we've collected

for q = 1:length(Qresult)
    
    filenm = [HOMEIMAGES '/' Qresult(q).annotation.folder '/' Qresult(q).annotation.filename];
    I = imread(filenm);

    for obj = 1:length(Qresult(q).annotation.object)
        data{i} = extract_patch(Qresult, q, obj, I, IMSIZE);

        % remove the annotation from D so we don't select it again!
        id = Qresult(q).annotation.object(obj).id;
        a = cell2mat({D(j(q)).annotation.object.id});
        struct_loc = find(a == id,1);
        D(j(q)).annotation.object = D(j(q)).annotation.object(setdiff(1:length(D(j(q)).annotation.object),struct_loc));
        
        % check to see if we've collected enough examples
        if i >= N
            return;
        end
        i = i + 1;
    end
end




%% ======================= GET_NEGATIVE_SAMPLES   =========================
function [data, D] = get_negative_samples(Qresult, N, HOMEIMAGES, IMSIZE, data, D, j, SHRINK_BORDER)
% sample N examples from non-queary locations in Qresult, extract a 
% bounding patch and place it in a cell {data}

i = 1;          % count the number of query matches we've collected
OVERLAP = .10;  % how much (%) negative sample can contain positive class

samples = sort(randsample(length(Qresult), N, 1));

for q = unique(samples)'

    filenm = [HOMEIMAGES '/' Qresult(q).annotation.folder '/' Qresult(q).annotation.filename];
    I = imread(filenm);
    
    % check to see if we already have constructed a non-query mask to
    % restrict the locations we sample from.
    if ~isfield(D(j(q)).annotation, 'nonmask')
        filenm2 = [HOMEIMAGES '/' Qresult(q).annotation.folder '/non' Qresult(q).annotation.filename];
        if exist(filenm2, 'file')
            D(j(q)).annotation.nonmask = imread(filenm2);
        else
            disp(['    forming a negative mask for ' filenm]); 
            D(j(q)).annotation.nonmask = makeNonMask(Qresult(q).annotation, I, SHRINK_BORDER);
            imwrite(D(j(q)).annotation.nonmask, filenm2, 'PNG');
        end
    end
    
    for k = 1:length(find(samples == q))   
        data{i} = sample_point(I, D(j(q)).annotation.nonmask, IMSIZE, OVERLAP);
        i = i + 1;
    end
end




%%  ========= get just the image patch containing the annotation ==========
function data = extract_patch(Qresult, q, obj, I, IMSIZE)

if length(Qresult(q).annotation.object(obj).polygon.pt) > 1
    x = cellfun(@str2num, {(Qresult(q).annotation.object(obj).polygon.pt.x)});
    y = cellfun(@str2num, {(Qresult(q).annotation.object(obj).polygon.pt.y)});    
    cmin = max(1,min(x));
    cmax = min(size(I,2),max(x));
    rmin = max(1,min(y));
    rmax = min(size(I,1),max(y));
    data = I(rmin:rmax, cmin:cmax);
else
    % if the data is a point instead of a contour, extract DATASETS.IMSIZE patch
    % around the point
    h = floor(IMSIZE(2)/2);
    w = floor(IMSIZE(1)/2);
   
    cmin = max(1, str2double(Qresult(q).annotation.object(obj).polygon.pt.x) -w);
    rmin = max(1, str2double(Qresult(q).annotation.object(obj).polygon.pt.y) -h);
    cmax = min(size(I,2),cmin + IMSIZE(2)-1);
    rmax = min(size(I,1),rmin + IMSIZE(1)-1);
    

    data = I(rmin:rmax, cmin:cmax);
end





%% ===== create a Mask of points containing NON-tag_string examples =======
function BW = makeNonMask(annotation, I, border)

% try to load a negative mask, if we cannot, then create it and save it

BW = logical(ones(size(I))); %#ok<LOGL>

for obj = 1:length(annotation.object)
    
    x = cellfun(@str2num, {(annotation.object(obj).polygon.pt.x)});
    y = cellfun(@str2num, {(annotation.object(obj).polygon.pt.y)});     
    BW = BW & ~bwmorph(poly2mask(x,y,size(I,1),size(I,2)), 'dilate', border);
end





%% =================== sample a valid data point ==========================
function data = sample_point(I, nonmask, IMSIZE, OVERLAP)

nonvalid = 1;

while nonvalid
    h = floor(IMSIZE(2)/2);
    w = floor(IMSIZE(1)/2);
    
    % sample a location 
    r = h+ceil((size(I,1)-2*h)*rand);
    c = w+ceil((size(I,2)-2*w)*rand);
    
    % get a patch around the location
    cmin = max(1, c-w);
    rmin = max(1, r-h);
    cmax = min(size(I,2),cmin+IMSIZE(2)-1);
    rmax = min(size(I,1),rmin+IMSIZE(1)-1);
    
    % check if it is valid
    if sum(sum(nonmask(rmin:rmax,cmin:cmax)))/prod(IMSIZE) > OVERLAP
        nonvalid = 0;
        data = I(rmin:rmax,cmin:cmax);
    end
end

% display for debugging what we sampled.
% figure(1);  axis image; hold on;  % imshow(I); 
% line([cmin cmax cmax cmin cmin], [rmin rmin rmax rmax rmin]);
