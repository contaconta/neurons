function  LEARNERS = p_EnumerateLearners(LEARNERS, IMSIZE)
%P_ENUMERATELEARNERS
%
%   LEARNERS = p_EnumerateLearners(LEARNERS, IMSIZE) parses the list of
%   weak learner types in LEARNERS.types, generates each individual weak
%   learner described by a string, and stores the strings in LEARNERS.list
%   as a cell. IMSIZE is the standard detector window size (contained in the
%   DATASETS structure).
%
%   Examples:
%   ----------------------
%   LEARNERS.types = {'HA_x1_y1'};
%   LEARNERS = p_EnumerateLearners(LEARNERS, [24 24]);
%
%   See also P_TRAIN, P_SETTINGS

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


tic; %disp('...defining the weak learners.');
LEARNERS.list = [];  LEARNERS.data = [];

for l=1:length(LEARNERS.types)
    switch LEARNERS.types{l}(1:2)
        
    case 'IT'
        [IT_LIST, IT_DATA] = enumerate_it(IMSIZE);
        LEARNERS.list = [LEARNERS.list IT_LIST'];
        LEARNERS.data = [LEARNERS.data IT_DATA'];


    otherwise
    learner_list = mexEnumerateLearners(LEARNERS.types, IMSIZE);
    LEARNERS.list = [LEARNERS.list learner_list];
    LEARNERS.data = [LEARNERS.data cell(1, length(learner_list))];
    % TODO: if features handled by the MEX need to store data in LEARNERS,
    % we need to adjust code to do this
    
    end
end


%LEARNERS.list = mexEnumerateLearners(LEARNERS.types, IMSIZE);

for i = 1:length(LEARNERS.types)
    type = LEARNERS.types{i};
    disp(['   defined ' type(1:2) ' learners.']);    
end

disp(['   Defined ' num2str(length(LEARNERS.list)) ' learners. Elapsed time ' num2str(toc) ' seconds.']);





%%-------------------------------------------------------------------------
function [IT_LIST, IT_DATA] = enumerate_it(IMSIZE)

d_cl = dir('./temp/Model-8-6000-3-i/FIB*.cl');
d_im = dir('./temp/Model-8-6000-3-i/FIB*.png');

counter = 1;
IT_LIST = {};
IT_DATA = {};

for i=1:length(d_cl)
    cloudpts = load(['./temp/Model-8-6000-3-i/' d_cl(i).name]);
    I = imread(['./temp/Model-8-6000-3-i/' d_im(i).name]);
    
    for j = 1:length(cloudpts)
        
        x = floor(cloudpts(j,1));
        y = floor(size(I,1) - cloudpts(j,2) - .001);
        if (x <= size(I,2)-floor(IMSIZE(1)/2)) && (y <= size(I,1)-floor(IMSIZE(2)/2)) && (x > floor(IMSIZE(1)/2)) && (y > floor(IMSIZE(2)/2))      
            IT_DATA{counter} = I(y-floor(IMSIZE(1)/2):y+floor(IMSIZE(1)/2), x-floor(IMSIZE(2)/2):x+floor(IMSIZE(2)/2)); %#ok<AGROW>
            IT_LIST{counter} = ['IT_' num2str(counter)]; %#ok<AGROW>
            counter = counter + 1;
        end
    end
    
end

