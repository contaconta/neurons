function v = l2hys(v, varargin)
%L2HYS
%
%   v = l2hys(v, threshold) - given a REAL vector, computes the L2-Hys (L2 norm
%   followed by clipping, followed by renomalization).  The clipping
%   threshold can be optinally specified as 2nd argument (default = 0.2).
%
%   example
%   bar(l2hys([100 1 203 24 245 33 234 23]))
%
%   Copyright © 2008 Kevin Smith
%
%   See also L2NORM

thresh = .2;

if nargin > 1
    thresh = varargin{1};
end

% normalize
v = l2norm(v);

% threshold
v(v > thresh) = thresh;

% re-normalize
v = l2norm(v);