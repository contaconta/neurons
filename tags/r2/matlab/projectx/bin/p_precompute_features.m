function p_precompute_features(SET, LEARNERS)
%
%   TODO: WRITE DOC

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

%tic;
disp('    Precomputing features on the TRAIN SET    ');
%cprintf('black', '  '); cprintf('-black', 'defined %d HA learners:', length(learner_list)); cprintf('black', ' %s\n', LEARNERS.types{l});
        

% start the memDaemon, which is used to store our precomputed feature responses
start_memDaemon(LEARNERS, SET);

W = wristwatch('start', 'end', length(LEARNERS.list), 'every', 5000);

% TODO: instead of storing response rows all at once, or one row at a time, it will be efficient to store chunks
% sized to fit in memory

for l = 1:length(LEARNERS.list)
    
    W = wristwatch(W, 'update', l, 'text', '       precomputed feature ');

    % precompute the feature responses for each example for learner l
    mexStoreResponse(p_get_feature_responses(SET, LEARNERS.list(l), LEARNERS.data(l)),'row',l,'HA');
    %responses = p_get_feature_responses(SET, LEARNERS.list(l), LEARNERS.data(l));
    %mexStoreResponse(responses,'row',l,'HA');
end

%toc; 










function start_memDaemon(LEARNERS, SET)
%%=========================================================================
dtype = get_datatype(LEARNERS.types);
system('killall memDaemon'); pause(0.1); system(['./bin/memDaemon ' int2str(length(SET.class)) ' ' int2str(length(LEARNERS.list)) ' ' dtype ' &']); pause(0.1);
[s,r]=system('ps -ef | grep memDaemon | grep -v -e "grep"');
if r
    disp(['       started memDaemon : ./bin/memDaemon ' int2str(length(SET.class)) ' ' int2str(length(LEARNERS.list)) ' ' dtype ' &']);
else
    disp('       memDaemon is not running'); 
end



%%=========================================================================
function dtype = get_datatype(types)

dtype = cell(size(types));

for t = 1:length(types)
    switch types{t}(1:2)
        case 'HA'
            dtype{t} = 'int';
        case 'IT'
            dtype{t} = 'int';
        case 'FR'
            dtype{t} = 'double';
        case '??'
            dtype{t} = 'int';
        otherwise
            error('Error p_precompute_features.m: could not find appropriate function for learner');
    end
end
            
dtype = unique(dtype);

if length(dtype) > 1
    error('Error p_precompute_features.m: attempted to mix data types in memdaemon');
else
    dtype = dtype{1};
end

