function SP = ada_spedge_define(varargin)
%
%   SP = ada_spedge_define(IMSIZE, ANGLES)
%
%   defines a spedge for each pixel and angle
%
%
%

IMSIZE = varargin{1};
angles = varargin{2};
sigmas = varargin{3};

% pre-allocated for speed
num_learners = prod(IMSIZE)*length(angles)*length(sigmas);
SP(num_learners).angles = [];
SP(num_learners).sigmas = [];
SP(num_learners).row = [];
SP(num_learners).col = [];
SP(num_learners).polarity = [];
SP(num_learners).theta = [];

for i = 1:IMSIZE(1)
    for j = 1:IMSIZE(2);
        for k = 1:length(angles);
            for s = 1:length(sigmas);

                % because we will flatten the spedge tensor into a vector 
                % later, we need to order the spedge features in the order 
                % the spedge vector will be ordered
                sp_index = single(sub2ind([length(angles) length(sigmas) IMSIZE(1) IMSIZE(2)], k,s,i,j));
                
                SP(sp_index).polarity = single(1);
                SP(sp_index).theta = 0;
                
                SP(sp_index).angle = angles(k);
                SP(sp_index).sigma = sigmas(s);
                SP(sp_index).row = i;
                SP(sp_index).col = j;
            end
        end
    end
end


%c_num = 1;


   %SP(c_num).sp_index = single(sub2ind([length(angles) length(sigmas) IMSIZE(1) IMSIZE(2)], k,s,i,j));
%                 SP(c_num).polarity = single(1);
%                 SP(c_num).theta = 0;
%                 
%                 SP(c_num).angle = angles(k);
%                 SP(c_num).sigma = sigmas(s);
%                 SP(c_num).row = i;
%                 SP(c_num).col = j;
% 
%                 c_num = c_num + 1;