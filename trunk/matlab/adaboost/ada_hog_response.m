function [f, HOG] = ada_hog_response(I, bin, cellc, cellr, orientationbins, cellsize, blocksize, varargin)
%
%
%
%
%

% if the HOG is provided, we don't need to compute it, just index it
if nargin > 7
    HOG = varargin{1};
else
    HOG = hog(I, 'orientationbins', orientationbins, 'cellsize', cellsize, 'blocksize', blocksize);
end

f = HOG(cellr, cellc, bin);


