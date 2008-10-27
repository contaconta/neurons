function HAAR = ada_haar_define(varargin)
%ADA_DEFINE_CLASSIFIERS defines a set of weak haar like classifiers.
%
%   REWRITE!!!!!
%   a = ada_haar_define(IMSIZE, 'type', {'haar1', 'haar2'});
%   a = 1x23423 struct with fields: descriptor, hinds, hvals, polarity, theta
%
%   WEAK = ada_define_classifiers(IMSIZE, ...) constructs a set of weak 
%   haar-like classifiers that define the difference between image regions  
%   [from Viola-Jones IJCV 2004].  IMSIZE = [WIDTH HEIGHT] is the standard 
%   size of the images used for training and testing. Optional argument
%   'types' followed by a vector (default [1 2 3 4 5]) specifies which haar-like
%   classifier types to use. Each type is defined by a vector of length 17:
%
%   vert2    horz2      vert3     horz3   checker   
%   ###--|   ######    ###---###  ######   ###--|   ######
%   ###  |   ######    ###   ###  |    |   ###  |   ##  ##
%   ###  |   |    |    ###   ###  |    |   |  ###   ##  ##
%   ###__|   |____|    ###---###  ######   |__###   ######
%                                                   (not implemented)
%
%   1:      type
%   2-3:    first white UL coordinate (x,y)
%   4-5:    first white LR coordinate (x,y)
%   6-7:    first black UL coordinate (x,y)
%   8-9:    first black LR coordinate (x,y)
%   10-11:  second white UL coordinate (x,y)
%   12-13:  second white LR coordinate (x,y)
%   14-15:  second black UL coordinate (x,y)
%   16-17:  second black LR coordinate (x,y)
%
%   Optional parameters: 'v' for verbose mode, 'disp' for a figure display
%   of the weak learners, 'H_LIMITS' followed by [H_MIN H_STEP H_MAX]
%   define the minimim/maximum/interval height that define the haar-like
%   features, 'W_LIMITS' followed by [W_MIN W_STEP W_MAX] for width. 
%   Returns WEAK, a struct containing an Nx18 descriptor where N is the 
%   number of weak classifiers.  
%
%   Example:  
%   Defines & plots weak classifiers in WEAK for a 13x13 image using haar 
%   types 2,4,5 with min horizontal size of 6 and max of 13 with 3 pixel steps.  
%   -----------------
%   WEAK = ada_define_classifiers([13 13], 'disp', 'type', [2,4,5], ...
%                                'H_LIMITS', [6 3 13]);
%
%   Copyright 2008 Kevin Smith
%
%   See also ADA_PLOT_HAAR_FEATURE


% define parameters
IMSIZE = varargin{1};  IMAGE_W = IMSIZE(1); IMAGE_H = IMSIZE(2);  V = 0; D = 0;
MAX_W = 15; MAX_H = 15; U_MIN_W = 1; U_MIN_H = 1; U_STEP_W = 1; U_STEP_H = 1;
BIG_SAFE_NUMBER = 100000;                           % large enough for IMSIZE = [24 24]
HAAR(BIG_SAFE_NUMBER).descriptor = [];
HAAR(BIG_SAFE_NUMBER).hinds = [];
HAAR(BIG_SAFE_NUMBER).hvals = [];
%HAAR(BIG_SAFE_NUMBER).theta = [];

%WEAK.descriptor = zeros([BIG_SAFE_NUMBER 17]);      % preallocate descriptor space
%WEAK.fast = zeros(BIG_SAFE_NUMBER, prod(IMSIZE));   % preallocate fast descriptor space

SHAPES = [1 2 3 4 5];
c_num = 1;                                  % the weak classifier index
tic; 


% handle optional arguments
for i = 2:nargin
    if strcmp(varargin{i}, 'shapes')
        SHAPES = varargin{i+1};
    end
    if strcmp(varargin{i}, 'v')
        V = 1;
    end
    if strcmp(varargin{i}, 'disp')
        D = 1; V = 1;
    end
    if strcmp(varargin{i}, 'H_LIMITS')
        LIMS = varargin{i+1};
        U_MIN_H = LIMS(1); U_STEP_H = LIMS(2); MAX_H = min(LIMS(3),IMAGE_H);
    end
    if strcmp(varargin{i}, 'W_LIMITS')
        LIMS = varargin{i+1};
        U_MIN_W = LIMS(1); U_STEP_W = LIMS(2); MAX_W = min(LIMS(3), IMAGE_W);
    end
end



%==========================================================================
% VERTICAL HAAR FEATURE
%==========================================================================

if ismember('vert2', SHAPES)
    MIN_W = max(2,quant(U_MIN_W,2));            	% min feature width
    MIN_H = U_MIN_H;                                % min feature height
    W_STEP = max(2,quant(U_STEP_W,2));              % feature width size step
    H_STEP = U_STEP_H;                              % feature height size step

    for w = MIN_W:W_STEP:MAX_W
        for h = MIN_H:H_STEP:MAX_H
            for y = 1:IMAGE_H - h + 1
                for x = 1:IMAGE_W - w + 1
                   
                    HAAR(c_num).descriptor(1) = 1;  % TYPE 1 = haar vertical
                    HAAR(c_num).descriptor(2:3) = [x y];
                    HAAR(c_num).descriptor(4:5) = [x+w/2-1 y+h-1];
                    HAAR(c_num).descriptor(6:7) = [x+w/2 y];
                    HAAR(c_num).descriptor(8:9) = [x+w-1 y+h-1];
            
                    if (x > 1) && (y > 1)
                        HAAR(c_num).hinds = [HAAR(c_num).hinds sub2ind(IMSIZE, y-1, x-1)];
                        HAAR(c_num).hvals = [HAAR(c_num).hvals -1];
                    end
                    if x > 1
                        HAAR(c_num).hinds = [HAAR(c_num).hinds sub2ind(IMSIZE, y+h-1, x-1)];
                        HAAR(c_num).hvals = [HAAR(c_num).hvals 1];
                    end
                    if y > 1
                        HAAR(c_num).hinds = [HAAR(c_num).hinds sub2ind(IMSIZE, y-1, x+w/2-1) sub2ind(IMSIZE, y-1, x+w-1)];
                        HAAR(c_num).hvals = [HAAR(c_num).hvals +2 -1];

                    end
                    
                    HAAR(c_num).hinds = [HAAR(c_num).hinds sub2ind(IMSIZE, y+h-1, x+w/2-1) sub2ind(IMSIZE, y+h-1, x+w-1)];
                    HAAR(c_num).hvals = [HAAR(c_num).hvals -2  1];
                    HAAR(c_num).polarity = 1;
                    HAAR(c_num).theta = 0;
                    
                    if D   %plot the classifier
                         ada_plot_haar_feature(WEAK.descriptor(c_num,:), IMSIZE);
                    end
                    c_num = c_num + 1;
                end
            end
        end
    end
    if V; disp(['...vert2 classifiers defined in ' num2str(toc) ' seconds']); tic; end;
end



%==========================================================================
% HORIZONTAL HAAR FEATURE
%==========================================================================

if ismember('horz2', SHAPES)
    MIN_W = U_MIN_W;                                % min feature width
    MIN_H = max(2,quant(U_MIN_H,2));                % min feature height
    W_STEP = U_STEP_W;                              % feature width size step
    H_STEP = max(2,quant(U_STEP_H,2));               % feature height size step

    for h = MIN_H:H_STEP:MAX_H
        for w = MIN_W:W_STEP:MAX_W      
            for y = 1:IMAGE_H - h +1
                for x = 1:IMAGE_W - w +1

                    HAAR(c_num).descriptor(1) = 2;  % TYPE 2 = haar horizontal
                    HAAR(c_num).descriptor(2:3) = [x y];
                    HAAR(c_num).descriptor(4:5) = [x+w-1 y+h/2-1];
                    HAAR(c_num).descriptor(6:7) = [x y+h/2];
                    HAAR(c_num).descriptor(8:9) = [x+w-1 y+h-1];
        
                    if (x > 1) && (y > 1)
                        HAAR(c_num).hinds = [HAAR(c_num).hinds sub2ind(IMSIZE, y-1, x-1)];
                        HAAR(c_num).hvals = [HAAR(c_num).hvals -1];
                    end
                    if x > 1
                        HAAR(c_num).hinds = [HAAR(c_num).hinds sub2ind(IMSIZE, y+h/2-1, x-1) sub2ind(IMSIZE, y+h-1,   x-1)];
                        HAAR(c_num).hvals = [HAAR(c_num).hvals 2 -1];
                    end
                    if y > 1
                        HAAR(c_num).hinds = [HAAR(c_num).hinds sub2ind(IMSIZE, y-1, x+w-1)];
                        HAAR(c_num).hvals = [HAAR(c_num).hvals 1];
                    end
                    
                    HAAR(c_num).hinds = [HAAR(c_num).hinds sub2ind(IMSIZE, y+h/2-1, x+w-1) sub2ind(IMSIZE, y+h-1, x+w-1)];
                    HAAR(c_num).hvals = [HAAR(c_num).hvals -2 1];
                    HAAR(c_num).polarity = 1;
                    HAAR(c_num).theta = 0;

                    if D % plot the classifier
                        ada_plot_haar_feature(WEAK.descriptor(c_num,:), IMSIZE);
                    end
                    c_num = c_num + 1;
                end
            end
        end
    end
    if V; disp(['...horz2 classifiers defined in ' num2str(toc) ' seconds']); tic; end;
end

%==========================================================================
% TRIPLE VERTICLE HAAR FEATURE
%==========================================================================

if ismember('vert3', SHAPES)
    MIN_W = max(3,quant(U_MIN_W,3));            	% min feature width
    MIN_H = U_MIN_H;                                % min feature height
    W_STEP = max(3,quant(U_STEP_W,3));              % feature width size step
    H_STEP = U_STEP_H;                              % feature height size step
    
    for w = MIN_W:W_STEP:MAX_W
        for h = MIN_H:H_STEP:MAX_H        
            for y = 1:IMAGE_H - h + 1
                for x = 1:IMAGE_W - w + 1
                   
                    HAAR(c_num).descriptor(1) = 3;  % TYPE 3 = 3-rectangle haar vertical
                    HAAR(c_num).descriptor(2:3) = [x y];
                    HAAR(c_num).descriptor(4:5) = [x+w/3-1 y+h-1];
                    HAAR(c_num).descriptor(6:7) = [x+w/3 y];
                    HAAR(c_num).descriptor(8:9) = [x+2*w/3-1 y+h-1];
                    HAAR(c_num).descriptor(10:11) = [x+2*w/3 y];
                    HAAR(c_num).descriptor(12:13) = [x+w-1 y+h-1];

                    
                    if (x > 1) && (y > 1)
                        HAAR(c_num).hinds = [HAAR(c_num).hinds sub2ind(IMSIZE, y-1, x-1)];
                        HAAR(c_num).hvals = [HAAR(c_num).hvals -1];
                    end
                    if x > 1
                        HAAR(c_num).hinds = [HAAR(c_num).hinds sub2ind(IMSIZE, y+h-1, x-1)];
                        HAAR(c_num).hvals = [HAAR(c_num).hvals 1];
                    end
                    if y > 1
                        HAAR(c_num).hinds = [HAAR(c_num).hinds sub2ind(IMSIZE, y-1, x+w/3-1) sub2ind(IMSIZE, y-1, x+2*w/3-1) sub2ind(IMSIZE, y-1, x+w-1)]; 
                        HAAR(c_num).hvals = [HAAR(c_num).hvals 2 -2 1];
                    end
                    
                    HAAR(c_num).hinds = [HAAR(c_num).hinds sub2ind(IMSIZE, y+h-1, x+w/3-1) sub2ind(IMSIZE, y+h-1, x+2*w/3-1) sub2ind(IMSIZE, y+h-1, x+w-1)];
                    HAAR(c_num).hvals = [HAAR(c_num).hvals -2 2 -1];
                    HAAR(c_num).polarity = 1;
                    HAAR(c_num).theta = 0;

                    if D % plot the classifier
                        ada_plot_haar_feature(WEAK.descriptor(c_num,:), IMSIZE);
                    end
                    c_num = c_num + 1;
                end
            end
        end
    end
    if V; disp(['...vert3 classifiers defined in ' num2str(toc) ' seconds']); tic; end;
end

%==========================================================================
% TRIPLE HORIZONTAL HAAR FEATURE
%==========================================================================

if ismember('horz3', SHAPES)
    MIN_W = U_MIN_W;                                % min feature width
    MIN_H = max(3,quant(U_MIN_H,3));                % min feature height
    W_STEP = U_STEP_W;                              % feature width size step
    H_STEP = max(3,quant(U_STEP_H,3));              % feature height size step

    for w = MIN_W:W_STEP:MAX_W
        for h = MIN_H:H_STEP:MAX_H        
            for y = 1:IMAGE_H - h + 1
                for x = 1:IMAGE_W - w + 1
                   
                    HAAR(c_num).descriptor(1) = 4;  % TYPE 4 = 3-rectangle haar horizontal
                    HAAR(c_num).descriptor(2:3) = [x y];
                    HAAR(c_num).descriptor(4:5) = [x+w-1 y+h/3-1];
                    HAAR(c_num).descriptor(6:7) = [x y+h/3];
                    HAAR(c_num).descriptor(8:9) = [x+w-1 y+2*h/3-1];
                    HAAR(c_num).descriptor(10:11) = [x y+2*h/3];
                    HAAR(c_num).descriptor(12:13) = [x+w-1 y+h-1];

                    if (x > 1) && (y > 1)
                        HAAR(c_num).hinds = [HAAR(c_num).hinds sub2ind(IMSIZE, y-1, x-1)];
                        HAAR(c_num).hvals = [HAAR(c_num).hvals -1];
                    end
                    if x > 1
                        HAAR(c_num).hinds = [HAAR(c_num).hinds sub2ind(IMSIZE, y+h/3-1, x-1) sub2ind(IMSIZE, y+2*h/3-1, x-1) sub2ind(IMSIZE, y+h-1, x-1)]; 
                        HAAR(c_num).hvals = [HAAR(c_num).hvals 2 -2 1];
                    end
                    if y > 1
                        HAAR(c_num).hinds = [HAAR(c_num).hinds sub2ind(IMSIZE, y-1, x+w-1)];
                        HAAR(c_num).hvals = [HAAR(c_num).hvals 1];
                    end

                    HAAR(c_num).hinds = [HAAR(c_num).hinds sub2ind(IMSIZE, y+h/3-1, x+w-1) sub2ind(IMSIZE, y+2*h/3-1, x+w-1) sub2ind(IMSIZE, y+h-1, x+w-1)];
                    HAAR(c_num).hvals = [HAAR(c_num).hvals -2 2 1];
                    HAAR(c_num).polarity = 1;
                    HAAR(c_num).theta = 0;

                    if D % plot the classifier
                        ada_plot_haar_feature(WEAK.descriptor(c_num,:), IMSIZE);
                    end
                    c_num = c_num + 1;
                end
            end
        end
    end
    if V; disp(['...horz3 classifiers defined in ' num2str(toc) ' seconds']); tic; end;
end


%==========================================================================
% 4-rectangle HAAR FEATURE
%==========================================================================

if ismember('checker', SHAPES)
    MIN_W = max(2,quant(U_MIN_W,2));                % min feature width
    MIN_H = max(2,quant(U_MIN_H,2));                % min feature height
    W_STEP = max(2,quant(U_STEP_W,2));              % feature width size step
    H_STEP = max(2,quant(U_STEP_H,2));              % feature height size step
 
    for w = MIN_W:W_STEP:MAX_W
        for h = MIN_H:H_STEP:MAX_H      
            for y = 1:IMAGE_H - h + 1
                for x = 1:IMAGE_W - w + 1
                   
                    HAAR(c_num).descriptor(1) = 5;  % TYPE 5 = haar 4-rectangle
                    HAAR(c_num).descriptor(2:3) = [x y];
                    HAAR(c_num).descriptor(4:5) = [x+w/2-1 y+h/2-1];
                    HAAR(c_num).descriptor(6:7) = [x+w/2 y];
                    HAAR(c_num).descriptor(8:9) = [x+w-1 y+h/2-1];
                    HAAR(c_num).descriptor(10:11) = [x+w/2 y+h/2];
                    HAAR(c_num).descriptor(12:13) = [x+w-1 y+h-1];
                    HAAR(c_num).descriptor(14:15) = [x y+h/2];
                    HAAR(c_num).descriptor(16:17) = [x+w/2-1 y+h-1];

                    if (x > 1) && (y > 1)
                        HAAR(c_num).hinds = [HAAR(c_num).hinds sub2ind(IMSIZE, y-1, x-1)];
                        HAAR(c_num).hvals = [HAAR(c_num).hvals -1];
                    end
                    if x > 1
                        HAAR(c_num).hinds = [HAAR(c_num).hinds sub2ind(IMSIZE, y+h/2-1, x-1) sub2ind(IMSIZE, y+h-1,   x-1)];
                        HAAR(c_num).hvals = [HAAR(c_num).hvals 2 -1];
                    end
                    if y > 1
                        HAAR(c_num).hinds = [HAAR(c_num).hinds sub2ind(IMSIZE, y-1, x+w/2-1) sub2ind(IMSIZE, y-1, x+w-1)];
                        HAAR(c_num).hvals = [HAAR(c_num).hvals 2 -1];
                    end
                    
                    HAAR(c_num).hinds = [HAAR(c_num).hinds sub2ind(IMSIZE, y+h/2-1, x+w/2-1) sub2ind(IMSIZE, y+h/2-1, x+w-1) sub2ind(IMSIZE, y+h-1, x+w/2-1) sub2ind(IMSIZE, y+h-1, x+w-1)];
                    HAAR(c_num).hvals = [HAAR(c_num).hvals -4 2 2 -1];
                    HAAR(c_num).polarity = 1;
                    HAAR(c_num).theta = 0;

                    % plot the classifier
                    if D
                        ada_plot_haar_feature(WEAK.descriptor(c_num,:), IMSIZE);
                    end
                    c_num = c_num + 1;
                end
            end
        end
    end
    if V; disp(['...checker classifiers defined in ' num2str(toc) ' seconds']); end;
end

% get rid of the wasted space from the memory allocation (was used to speed things up)
LAST_CLASSIFIER = c_num - 1;
HAAR = HAAR(1:LAST_CLASSIFIER);
