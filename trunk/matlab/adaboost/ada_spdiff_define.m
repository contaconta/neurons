function spdiff = ada_spdiff_define(varargin)
%
%   spdiff = ada_spdiff_define(IMSIZE, ANGLES)
%
%   defines a spoke difference for each pixel and angle
%
%
%

IMSIZE = varargin{1};
angles = varargin{2};
sigmas = varargin{3};


count = 1;

combos = combnk(angles, 2);
            
for c = 1:size(combos,1);
    for i = 1:IMSIZE(1)
        for j = 1:IMSIZE(2);
            for s = 1:length(sigmas);

            spdiff(count).vec_index1 = single(sub2ind([length(angles) length(sigmas) IMSIZE(1) IMSIZE(2)], find(angles == combos(c,1),1),s,i,j));
            spdiff(count).vec_index2 = single(sub2ind([length(angles) length(sigmas) IMSIZE(1) IMSIZE(2)], find(angles == combos(c,2),1),s,i,j));
            
            spdiff(count).polarity = single(1);
            spdiff(count).theta = 0;

            spdiff(count).angle1 = combos(c,1);
            spdiff(count).angle2 = combos(c,2);
            spdiff(count).sigma = sigmas(s);
            spdiff(count).row = i;
            spdiff(count).col = j;

            count = count + 1;
            end
        end
    end
end



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