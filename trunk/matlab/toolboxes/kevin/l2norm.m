function v = l2norm(v)
%L2NORM
%
%   v = l2norm(v, threshold) - given a REAL vector, computes the L2 norm.  
%
%   example
%   bar(l2norm([100 1 203 24 245 33 234 23]))
%
%   Copyright © 2008 Kevin Smith
%
%   See also L2HYS

eta2 = 1e-10;

v = v / sqrt( sum(sum(sum(v.^2))) + eta2);
