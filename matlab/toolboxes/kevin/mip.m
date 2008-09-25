function I = mip(V, varargin)
%MIP Maximum/Minimum Intensity Projection 
%
%   I = mip(I, ...) - creates a maximum or minimum intensity projection 
%   image from a color image or a stack of images.  The projection can be 
%   made along an axis defined by the user as  'x', 'y', or 'z' (default).  
%   Optional arguments 'max' (default) and 'min' specify if the projection
%   contains maximum intensity values or minimum intensity values.  Option
%   'scale' scales the MIP to cover the range of intensities.
%
%   Example:
%   ------------
%   load mristack;
%   I = mip(mristack, 'z', 'max');
%   figure; imshow(I);
%   
%   Copyright 2008 Kevin Smith
%
%   See also GRAY2RGB, IMOVERLAY, IMADD, IMLINCOMB, MIP, MAT2GRAY

% the axis on which to make the MIP
axi = 3;
mode = 1;
sc = 0;

if nargin > 1;
    for i = 2:nargin-1
        switch varargin{i}
            case 'x'
                axi = 1;

            case 'y'
                axi = 2;

            case 'z'
                axi = 3;
            case 'max'
                mode = 1;
            case 'min'
                mode = 0;
            case 'scale'
                sc = 1;
        end
    end
end


if mode
    I = max(V, [ ], axi);
else
    I = min(V, [ ], axi);
end

if sc
    I = imadjust(I, stretchlim(I), []);
end


