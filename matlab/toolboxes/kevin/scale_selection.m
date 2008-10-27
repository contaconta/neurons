function SCALES = scale_selection(I, IMSIZE, varargin)
%SCALE_SELECTION automatically selects scales to scan an image detector
%
%   SCALES = scale_selection(I, IMSIZE)  
%   SCALES = scale_selection(I, IMSIZE, 'scale_factor', SCALE_FACTOR, 'limits', LIMITS)
%
%   SCALES contains scales TO RESIZE THE IMAGE.  The effective detector
%   scale is 1/scale.  LIMITS is given in detector scale size.  For
%   example,
%
%   scale_selection(zeros(100,100), [24 24], 'limits', [1 2])
%
%   returns   [0.5120    0.6400    0.8000    1.0000]
%
%   This effectively scales the detector from its original size to double
%   that size.
%   
%   Copyright © 2008 Kevin Smith
%
%   See also 

%% set default values
SCALE_FACTOR    = 1.25;
MIN_SCALE       = 1;
MAX_SCALE       = max([size(I,1)/IMSIZE(1) size(I,2)/IMSIZE(2)]);

%% handle optional arguments
if nargin > 2
    for i = 1:nargin-2
        if strcmp('scale_factor', varargin{i})
            SCALE_FACTOR = varargin{i+1};
        end
        if strcmp('limits', varargin{i})
            LIMITS = varargin{i+1};
            %MAX_SCALE = 1/LIMITS(1);
            %MIN_SCALE = 1/LIMITS(2);
            MIN_SCALE = LIMITS(1);
            MAX_SCALE = LIMITS(2);
        end
    end
end

%% create the list of scales 

SCALES = MIN_SCALE;
while SCALES(length(SCALES)) *SCALE_FACTOR < MAX_SCALE
    SCALES(length(SCALES) + 1) = SCALES(length(SCALES)) * SCALE_FACTOR;
end

SCALES = sort(1./SCALES);

