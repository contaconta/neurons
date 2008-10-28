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


c_num = 1;

for i = 1:IMSIZE(1)
    for j = 1:IMSIZE(2);
        for k = 1:length(angles);
            for s = 1:length(sigmas);
                % [angle, r, c]
                %SP(c_num).descriptor = [k i j];
                SP(c_num).index = sub2ind([length(angles) length(sigmas) IMSIZE(1) IMSIZE(2)], k,s,i,j);
                SP(c_num).polarity = 1;
                SP(c_num).theta = 0;

                c_num = c_num + 1;
            end
        end
    end
end
