function DATASETS = load_labelme_index(DATASETS)
%% LOAD_LABELME_INDEX
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


% check to see if we have already created an index of folders needed
filenm = ['./temp/' 'LMINDEX_' cell2mat(DATASETS.LABELME_FOLDERS) '.mat'];

if exist(filenm, 'file')
    % folders have been indexed, load from a file to save time indexing
    disp(['... loading LabelMeIndex for folders: ' cell2mat(DATASETS.LABELME_FOLDERS)]); pause(.001);
    D = load(filenm);
    DATASETS.LabelMeIndex = D.D;
else
    % we don't have these folders indexed, make new one & store for future
    D = LMdatabase(DATASETS.HOMEANNOTATIONS, DATASETS.LABELME_FOLDERS);
    DATASETS.LabelMeIndex = D;
    save(filenm, 'D');
end


