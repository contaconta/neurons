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

tic;
disp('Precomputing features on the TRAIN SET');

system('killall memDaemon');
system(['./bin/memDaemon ' int2str(length(SET.class)) ' ' int2str(length(LEARNERS.list)) ' int &']);
[s,r]=system('ps -ef | grep memDaemon | wc -l');
if (r ~= 3)
disp('memDaemon is not running');
else
disp(['started memDaemon : ./bin/memDaemon ' int2str(length(SET.class)) ' ' int2str(length(LEARNERS.list)) ' int &']);
end

W = wristwatch('start', 'end', length(LEARNERS.list), 'every', 5000);

% TEMP : Preload images for IT features
dirIm='/localhome/aurelien/usr/share/Data/LabelMe/Images/FIBSLICE/';
listImageNames = dir([dirIm 'FIB*']);
listImages = {};
for l=1:10 %length(listImageNames)
  listImages{l} = imread([dirIm listImageNames(l).name]);
end

loopOverLearners = true;

% FIXME : instead of storing response rows all at once, or one row at a time, it will be efficient to store chunks
% sized to fit in memory
if loopOverLearners == true
  for l = 1:length(LEARNERS.list)
    
    % switch LEARNERS.list(l)(1:2)
    W = wristwatch(W, 'update', l, 'text', '       precomputed feature ');
    
    % precompute the feature responses for each example for learner l
    %responses = uint32(p_get_feature_responses(SET, LEARNERS.list(l)));
    %disp('MATLAB p_get_feature_responses')
    learner_id = regexp(LEARNERS.list(l), '\d*', 'match');
    LEARNERS.data{l} = listImages{ceil((str2num(learner_id{1}{1})+1)/600)};
    responses = p_get_feature_responses(SET, LEARNERS.list(l),LEARNERS.data(l));
    
    %keyboard;
    
    % store the responses as a row vector
    %disp(['MATLAB mexStoreResponse ' num2str(l) ' ' num2str(size(responses))]);
    mexStoreResponse(responses,'row',l,'HA');
  end
else
  
  responses = p_get_feature_responses(SET, LEARNERS.list,listImages);
  %disp(['MATLAB mexStoreResponse ' num2str(length(responses))]);
  size(responses)
  for l = 1:size(responses,2)
    mexStoreResponse(responses(:,l),'row',l,'HA');
  end
end
toc;
