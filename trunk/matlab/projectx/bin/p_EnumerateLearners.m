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

LEARNERS.list = mexEnumerateLearners(LEARNERS.types, IMSIZE);

for i = 1:length(LEARNERS.types)
    type = LEARNERS.types{i};
    disp(['   defined ' type(1:2) ' learners.']);    
end

% LEARNERS.class = char(zeros([length(LEARNERS.list),2]));
% for i = 1:length(LEARNERS.list)
%     learner_string = LEARNERS.list{i};
%     LEARNERS.class(i,:) = learner_string(1:2);
% end

disp(['   Defined ' num2str(length(LEARNERS.list)) ' learners. Elapsed time ' num2str(toc) ' seconds.']);