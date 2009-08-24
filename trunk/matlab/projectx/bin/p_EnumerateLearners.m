function  LEARNERS = p_EnumerateLearners(LEARNERS, DATASETS)
%P_ENUMERATELEARNERS
%
%   LEARNERS = p_EnumerateLearners(LEARNERS, DATASETS) parses the list of
%   weak learner types in LEARNERS.types, generates each individual weak
%   learner described by a string, and stores the strings in LEARNERS.list
%   as a cell. DATASETS is a structure containing information about the
%   database.
%
%   Examples:
%   ----------------------
%   DATASETS.IMSIZE = [24 24];
%   LEARNERS.types = {'HA_x1_y1'};
%   LEARNERS = p_EnumerateLearners(LEARNERS, DATASETS);
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
        [IT_LIST, IT_DATA] = enumerate_it(DATASETS.IMSIZE);
        LEARNERS.list = [LEARNERS.list IT_LIST];
        LEARNERS.data = [LEARNERS.data IT_DATA];

    case 'FR'
        % parse the LEARNERS.types string
        %Nfrags = 1000;  xystep =2; smin = [4 4]; smax = [22 22]; ss = 2; p = 1;
        m = regexp(LEARNERS.types{l}, 'FR_N_(\d*)?_smin_(\d*)?_smax_(\d*)?_p_(\d*)', 'tokens');
        N = str2double(m{1}{1}); 
        smin = [str2double(m{1}{2}) str2double(m{1}{2})];
        smax = [str2double(m{1}{3}) str2double(m{1}{3})];
        p = str2double(m{1}{4}); 
        xystep = 2; ss = 2;
        
        [FR_LIST, FR_DATA] = enumerate_fr(N, xystep, smin, smax, ss, p, DATASETS);
        LEARNERS.list = [LEARNERS.list FR_LIST];
        LEARNERS.data = [LEARNERS.data FR_DATA];
        
     case 'HA'
        learner_list = mexEnumerateLearners(LEARNERS.types(l), DATASETS.IMSIZE);
        LEARNERS.list = [LEARNERS.list learner_list'];
        LEARNERS.data = [LEARNERS.data cell(1, length(learner_list))];
        
    otherwise
        error('Error p_EnumerateLearners: learners were not properly enumerated.');
        % TODO: if features handled by the MEX need to store data in LEARNERS,
        % we need to adjust code to do this
    
    end

end


for i = 1:length(LEARNERS.types)
    type = LEARNERS.types{i};
    disp(['   defined ' type(1:2) ' learners.']);    
end

%disp(['   Defined ' num2str(length(LEARNERS.list)) ' learners. Elapsed time ' num2str(toc) ' seconds.']);



%%-------------------------------------------------------------------------
function [FR_LIST, FR_DATA] = enumerate_fr(N, xystep, smin, smax, ss, p, DATASETS) %#ok<INUSL>
% N = the number of fragment images to use
% xystep = x & y step size, placing fragment in mask
% smin = min size, placing fragment in mask
% smax = max size, placing fragment in mask
% ns   = scale step size
% fsize = standard fragment size, to be stored in learner_data

if (DATASETS.IMSIZE(1) < smax(1) || DATASETS.IMSIZE(2) < smax(2))
    error('p_EnumerateLearners:enumerate_fr: the maximum fragment size has been defined as greater than the detector window');
end

count = 1;
nfragments = 1;

% create a vector containing the scales fragments will appear
s = smin:ss:smax;

% create a vector containing images we will extract the data from.
imglist = sort(ceil(length(DATASETS.LabelMeIndex)*rand([1 N])));

filenm = [];

% extract N fragments of size FSIZE from the data set
for n = 1:N
    newfilenm = [DATASETS.HOMEIMAGES '/'  DATASETS.LabelMeIndex(imglist(n)).annotation.folder '/'   DATASETS.LabelMeIndex(imglist(n)).annotation.filename];
    
    % load the image we will extract a fragment from, if not already loaded
    if ~strcmp(newfilenm, filenm)
        I = imread(newfilenm);
    end
    filenm = newfilenm;
    
    % extract a detector-sized sample
    r = max(1, ceil(size(I,1)*rand - DATASETS.IMSIZE(1)));
    c = max(1, ceil(size(I,2)*rand - DATASETS.IMSIZE(2)));
    sample = I(r:r+DATASETS.IMSIZE, c:c+DATASETS.IMSIZE);
    
    % extact the fragment
    fsize = randsample(s,1);
    r = max(1, ceil((size(sample,1)- fsize-1)*rand));
    c = max(1, ceil((size(sample,2)- fsize-1)*rand));
    fragment = sample(r:r+fsize-1, c:c+fsize-1);
    nfragments = nfragments + 1;
    
    BB = [c r  size(fragment,2)-1 size(fragment,1)-1];
    learner_str = ['FR_' num2str(count) '_p_' num2str(p) '_img_' num2str(count) '_bb_' num2str(BB(1)) '_' num2str(BB(2)) '_' num2str(BB(3)) '_' num2str(BB(4))];

    FR_LIST{count} = learner_str; %#ok<AGROW>
    FR_DATA{count} = fragment; %#ok<AGROW>

    count = count + 1;
    
%     % extact the fragment
%     fsize = randsample(s,1, true, s/sum(s) );
%     r = max(1, ceil(size(I,1)*rand - fsize));
%     c = max(1, ceil(size(I,2)*rand - fsize));
%     fragment = I(r:r+fsize-1, c:c+fsize-1);
%     nfragments = nfragments + 1;
%     
%     % TODO: here we could scale the fragment if we chose to
%     
%     
%     % loop through the BB locations
%     for r=1:xystep:DATASETS.IMSIZE - size(fragment,1)
%         for c=1:xystep:DATASETS.IMSIZE - size(fragment,2)
%             % BB is a 4-element vector with the form [XMIN YMIN WIDTH HEIGHT]; 
%             % these values are specified in spatial coordinates, so the 
%             % size of the scaled fragment will be WIDTH+1 HEIGHT+1.
%             
%             BB = [c r  size(fragment,2) size(fragment,1)];
%             
%             learner_str = ['FR_' num2str(count) '_p_' num2str(p) '_img_' num2str(count) '_bb_' num2str(BB(1)) '_' num2str(BB(2)) '_' num2str(BB(3)) '_' num2str(BB(4))];
%             
%             FR_LIST{count} = learner_str;
%             FR_DATA{count} = fragment;
%             
%             count = count + 1;
%         end
%     end
    
end

disp(['   defined ' num2str(count-1) ' FR learners from ' num2str(nfragments-1) ' fragments']);


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

