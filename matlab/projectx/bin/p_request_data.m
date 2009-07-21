function [data, D] = p_request_data(tag_string, N, varargin)
%% P_REQUEST_DATA
%
%   [data, D] = p_request_data('query', N, ...) requests N examples from
%   a LabelMe database selected using the query string. It returns DATA, a 
%   cell containing the N examples as well as D, the LabelMe index
%   containing query results which have not yet been requested.  To request
%   a previously unseen example, pass D as an optional argument.
%       
%   Example:
%   -----------
%   [data, D] = p_request_data('car', 1);  % request 1 car
%   [data, D] = p_request_data('car', 3, D);  % request the next 3 cars
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

HOMEIMAGES = '/osshare/Work/Data/LabelMe/Images';
HOMEANNOTATIONS = '/osshare/Work/Data/LabelMe/Annotations';

SAMP = 0;   % SAMP = 1 when we randomly sample from within an annotation, otherwise take the whole annotation

for k = 1:nargin-2
    switch class(varargin{k})
        case 'struct'
            D = varargin{k};

        case 'char'
            % TODO: HANDLE RANDOM SAMPLING HERE
            if strcmp(varargin{k}, 'sample')
                SAMP = 1;
                IMSIZE = varargin{k+1};
            end
    end
end

% create a new LabelMe index if it hasn't been passed as an argument
if ~exist('D', 'var')
    D = LMdatabase(HOMEANNOTATIONS);
end
   

[Qresult, j] = LMquery(D, 'object.name', tag_string);
if isempty(Qresult); error(['p_request_data: LabelMe cannot find any more examples of type ' tag_string ]);end

i = 1;              % count the number of query matches we've collected
data = cell(1,N);     % image example data will be stored in a cell


% pick out N examples from Qresult, and remove them from D so they are not
% re-selected. extract a bounding patch and place it in a cell {data}
for q = 1:length(Qresult)
    
    filenm = [HOMEIMAGES '/' Qresult(q).annotation.folder '/' Qresult(q).annotation.filename];
    I = imread(filenm);
    
    % TEMPORARY - MUST PROPERLY FORMAT THE DATA!
    I = mat2gray(I);
    
    for obj = 1:length(Qresult(q).annotation.object)
        
        if SAMP
            data{i} = sample_within_patch(Qresult, q, obj, I, IMSIZE);
        else
            data{i} = extract_patch(Qresult, q, obj, I);
        end

        % remove the annotation from D so we don't select it again!
        if ~SAMP
            id = Qresult(q).annotation.object(obj).id;
            a = cell2mat({D(j(q)).annotation.object.id});
            struct_loc = find(a == id,1);
            D(j(q)).annotation.object = D(j(q)).annotation.object(setdiff(1:length(D(j(q)).annotation.object),struct_loc));
        end
        
        % check to see if we've collected enough examples
        if i >= N
            return;
        end
        i = i + 1;
    end
end



%% get just the image patch containing the annotation
function data = extract_patch(Qresult, q, obj, I)

x = cellfun(@str2num, {(Qresult(q).annotation.object(obj).polygon.pt.x)});
y = cellfun(@str2num, {(Qresult(q).annotation.object(obj).polygon.pt.y)});    
cmin = max(1,min(x));
cmax = min(size(I,2),max(x));
rmin = max(1,min(y));
rmax = min(size(I,1),max(y));
data = I(rmin:rmax, cmin:cmax);

