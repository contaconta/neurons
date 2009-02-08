
function [f, EDGE] = ada_spdiff_response(angle1,angle2,sigma,row,col,varargin)

% we can save computations by passing a previously computed EDGE for the
% given sigma
if nargin > 6
    EDGE = varargin{1};
    [d1 EDGE]   = single_spedge(angle1, sigma, row, col, EDGE, 'edge');
    [d2]        = single_spedge(angle2, sigma, row, col, EDGE, 'edge');
    f = d1 - d2;
else
    I = varargin{1};
    [d1 EDGE]   = single_spedge(angle1, sigma, row, col, I);
    [d2]        = single_spedge(angle2, sigma, row, col, I);
    f = d1 - d2;
end

