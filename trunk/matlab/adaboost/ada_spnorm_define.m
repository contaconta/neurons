function SP = ada_spnorm_define(varargin)
%
%   SP = ada_spnorm_define(IMSIZE, ANGLES, STRIDE, EDGE_METHODS)
%
%   defines a spnorm for each pixel and angle
%
%
%

IMSIZE = varargin{1};
angles = varargin{2};
stride = varargin{3};
edge_methods = varargin{4};

% pre-allocated for speed
num_learners = ceil(IMSIZE(1)/stride)*ceil(IMSIZE(2)/stride)*length(angles)*length(edge_methods);
SP(num_learners).angles = [];
SP(num_learners).edge_method = [];
SP(num_learners).row = [];
SP(num_learners).col = [];
SP(num_learners).polarity = [];
SP(num_learners).theta = [];
SP(num_learners).stride = [];

for i = 1:ceil(IMSIZE(1)/stride)
    for j = 1:ceil(IMSIZE(2)/stride);
        for k = 1:length(angles);
            for s = 1:length(edge_methods);

                % because we will flatten the spnorm tensor into a vector 
                % later, we need to order the spnorm features in the order 
                % the spedge vector will be ordered
                sp_index = single(sub2ind([length(angles) length(edge_methods) ceil(IMSIZE(1)/stride) ceil(IMSIZE(2)/stride)], k,s,i,j));
                
                SP(sp_index).polarity = single(1);
                SP(sp_index).theta = 0;
                
                SP(sp_index).angle = angles(k);
                SP(sp_index).edge_method = edge_methods(s);
                SP(sp_index).row = i;
                SP(sp_index).col = j;
                SP(sp_index).stride = stride;
            end
        end
    end
end