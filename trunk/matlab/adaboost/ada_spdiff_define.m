function spdiff = ada_spdiff_define(varargin)
%
%   spdiff = ada_spdiff_define(IMSIZE, ANGLES, STRIDE, EDGE_METHODS)
%
%   defines a spoke difference for each pixel and angle
%
%
%

IMSIZE = varargin{1};
angles = varargin{2};
stride = varargin{3};
edge_methods = varargin{4};


count = 1;


% combnk enumerates combinations - we use this to find all pairs of angles
combos = combnk(angles, 2);

% pre-allocated for speed
num_learners = (IMSIZE(1)/stride)*(IMSIZE(2)/stride)*size(combos,1)*length(edge_methods);
spdiff(num_learners).angle1 =[];
spdiff(num_learners).angle2 = [];
spdiff(num_learners).edge_method = [];
spdiff(num_learners).row = [];
spdiff(num_learners).col = [];
spdiff(num_learners).polarity = [];
spdiff(num_learners).theta = [];
spdiff(num_learners).stride = [];

for c = 1:size(combos,1);
    for i = 1:ceil(IMSIZE(1)/stride)
        for j = 1:ceil(IMSIZE(2)/stride);
            for s = 1:length(edge_methods);

            spdiff(count).vec_index1 = sub2ind([length(angles) length(edge_methods) ceil(IMSIZE(1)/stride) ceil(IMSIZE(2)/stride)], find(angles == combos(c,1),1),s,i,j);
            spdiff(count).vec_index2 = sub2ind([length(angles) length(edge_methods) ceil(IMSIZE(1)/stride) ceil(IMSIZE(2)/stride)], find(angles == combos(c,2),1),s,i,j);
            
            spdiff(count).polarity = single(1);
            spdiff(count).theta = 0;

            spdiff(count).angle1 = combos(c,1);
            spdiff(count).angle2 = combos(c,2);
            spdiff(count).edge_method = edge_methods(s);
            spdiff(count).row = i;
            spdiff(count).col = j;
            
            spdiff(count).stride = stride;
            
            count = count + 1;
            end
        end
    end
end
