function hog = ada_hog_define(varargin)
%
%   hog = ada_hog_define(IMSIZE, bins, cellsize, blocksize)
%
%   defines a spoke difference for each pixel and angle
%
%
%

IMSIZE      = varargin{1};
bins        = varargin{2};
cellsize    = varargin{3};
%blocksize   = varargin{4};


numcells = IMSIZE./cellsize;

num_learners = prod(numcells)*bins;
hog(num_learners).cellr =[];
hog(num_learners).cellc = [];
hog(num_learners).oind = [];
hog(num_learners).orientation = [];
hog(num_learners).polarity = [];
hog(num_learners).theta = [];

angles = rad2deg(0: pi/(bins-1) :pi);


for r = 1:numcells(1)
    for c = 1:numcells(2)
        for o = 1:bins
            
            hog_ind = single(sub2ind([numcells(1) numcells(2) bins], r,c,o));
            
            hog(hog_ind).cellr = r;
            hog(hog_ind).cellc = c;
            hog(hog_ind).oind  = o;
            hog(hog_ind).orientation = angles(o);
            
            
            
            hog(hog_ind).polarity   = single(1);
            hog(hog_ind).theta      = 0;
            
        end
    end
end



% count = 1;
% 
% % combnk enumerates combinations - we use this to find all pairs of angles
% combos = combnk(angles, 2);
%             
% for c = 1:size(combos,1);
%     for i = 1:IMSIZE(1)
%         for j = 1:IMSIZE(2);
%             for s = 1:length(sigmas);
% 
%             spdiff(count).vec_index1 = single(sub2ind([length(angles) length(sigmas) IMSIZE(1) IMSIZE(2)], find(angles == combos(c,1),1),s,i,j));
%             spdiff(count).vec_index2 = single(sub2ind([length(angles) length(sigmas) IMSIZE(1) IMSIZE(2)], find(angles == combos(c,2),1),s,i,j));
%             
%             spdiff(count).polarity = single(1);
%             spdiff(count).theta = 0;
% 
%             spdiff(count).angle1 = combos(c,1);
%             spdiff(count).angle2 = combos(c,2);
%             spdiff(count).sigma = sigmas(s);
%             spdiff(count).row = i;
%             spdiff(count).col = j;
%             
%             count = count + 1;
%             end
%         end
%     end
% end
% 


% for i = 1:IMSIZE(1)
%     for j = 1:IMSIZE(2);
%         for k = 1:length(angles);
%             for s = 1:length(sigmas);
% 
%                 % because we will flatten the spedge tensor into a vector 
%                 % later, we need to order the spedge features in the order 
%                 % the spedge vector will be ordered
%                 sp_index = single(sub2ind([length(angles) length(sigmas) IMSIZE(1) IMSIZE(2)], k,s,i,j));
%                 
%                 spdiff(sp_index).polarity = single(1);
%                 spdiff(sp_index).theta = 0;
%                 
%                 spdiff(sp_index).angle1 = angles(k);
%                 spdiff
%                 spdiff(sp_index).sigma = sigmas(s);
%                 spdiff(sp_index).row = i;
%                 spdiff(sp_index).col = j;
%             end
%         end
%     end
% end