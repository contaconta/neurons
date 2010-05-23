function [ROWS, COLS, INDS, POLS] = generate_viola_jones_features(varargin)
%GENERATE_VIOLA_JONES_FEATURES defines a set of weak haar like features.
%
%   [ROWS, COLS, INDS, POLS] = generate_viola_jones_features(SIZE, options)
%   constructs a set of haar-like features defined by the difference 
%   between image regions [as in Viola-Jones IJCV 2004]. SIZE = [WIDTH,
%   HEIGHT] defines the window bounding the haar-like features. 
%
%   OPTIONS:
%   'shapes', {'vert2', ...}
%   This option specifies which haar-like primitives to include. The
%   following primitives are available:
%
%   vert2    horz2      vert3     horz3   checker   
%   ###--|   ######    ###---###  ######   ###--|   
%   ###  |   ######    ###   ###  |    |   ###  |  
%   ###  |   |    |    ###   ###  |    |   |  ###  
%   ###__|   |____|    ###---###  ######   |__###  
%                    
%   Example: 
%   generate_viola_jones_features([12 12], 'shapes', {'vert2','horz3'});
%
%   'disp' - 
%   Visualizes the features as they are defined.
%   
%   'H_LIMITS' followed by [H_MIN H_STEP H_MAX] define the minimim/maximum/
%   interval height that define the haar-like features, 
%
%   'W_LIMITS' followed by [W_MIN W_STEP W_MAX] for width. 
%   
%   Returns:
%
%   ROWS - a cell containing row coordinates of points 1,2,3,4 for each
%          rectangle in the haar feature
%   COLS = a cell containing col coordinates of points 1,2,3,4 for each
%          rectangle in the haar feature
%   INDS - a cell containing linear indexes of points 1,2,3,4 for each 
%          rectangle in the haar feature
%   POLS - a matrix containing polarities of the rectangles in the haar
%          feature.
%   
%   1----2
%   |   ||
%   |---X|
%   3----4
%
%   Note that point X is the lower-right bound of the rectangle. Point 4
%   = X+[1,1], the row of point 3 is X(1)+1, the col of point 4 is X(2)+1.
%   This accounts for 'outer' integral images which contain 1 extra row and
%   column than the original image.
%
%   Example:  
%   Defines & visualizes vert2 and checker features for a 13x13 patch with min horizontal 
%   size of 6 and max of 13 with 3 pixel steps.  
%   -----------------
%   [R, C, I, P] = generate_viola_jones_features([13 13], 'disp', ...
%               'shapes', {'vert2', 'checker'}, 'H_LIMITS', [6 3 13]);
%
%   Copyright 2010 Kevin Smith
%
%   See also ADA_PLOT_HAAR_FEATURE


% define parameters
IMSIZE = varargin{1};  IMAGE_W = IMSIZE(2); IMAGE_H = IMSIZE(1);  V = 0; D = 0;
MAX_W = IMSIZE(2); MAX_H = IMSIZE(1); U_MIN_W = 1; U_MIN_H = 1; U_STEP_W = 1; U_STEP_H = 1;
SCAN_R_STEP = 1; SCAN_C_STEP = 1;
BIG_SAFE_NUMBER = 800000;                           % large enough for IMSIZE = [24 24]

INDS = cell(BIG_SAFE_NUMBER, 1);
POLS = cell(BIG_SAFE_NUMBER, 1);
ROWS = cell(BIG_SAFE_NUMBER, 1);
COLS = cell(BIG_SAFE_NUMBER, 1);

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
    if strcmp(varargin{i}, 'SCAN_X_STEP')
        SCAN_C_STEP = varargin{i+1};
    end
    if strcmp(varargin{i}, 'SCAN_Y_STEP')
        SCAN_R_STEP = varargin{i+1};
    end
end

if ~exist('SHAPES', 'var')
    SHAPES = {'vert2', 'vert3', 'horz2', 'horz3', 'checker'};
end

if D
    I = zeros(IMSIZE);
end

IISIZE = IMSIZE + [ 1 1];

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
            for r = 1:SCAN_R_STEP:IMAGE_H - h + 1
                for c = 1:SCAN_C_STEP:IMAGE_W - w + 1
                    
                    
                    R1 = [r r r+h r+h];
                    C1 = [c c+w/2 c c+w/2];
                    R2 = [r r r+h r+h];
                    C2 = [c+w/2 c+w c+w/2 c+w];

                    I1 = sub2ind(IISIZE, R1, C1);
                    I2 = sub2ind(IISIZE, R2, C2);
                    
                    ROWS{c_num} = {R1, R2};
                    COLS{c_num} = {C1, C2};
                    INDS{c_num} = {I1, I2};
                    POLS{c_num} = [1 -1];

                    
                    if D
                        rect_vis(I,ROWS{c_num},COLS{c_num}, POLS{c_num});
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
            for r = 1:SCAN_R_STEP:IMAGE_H - h +1
                for c = 1:SCAN_C_STEP:IMAGE_W - w +1

                    R1 = [r r r+h/2 r+h/2];
                    C1 = [c c+w c c+w];
                    R2 = [r+h/2 r+h/2 r+h r+h];
                    C2 = [c c+w c c+w];
                    
                    I1 = sub2ind(IISIZE, R1, C1);
                    I2 = sub2ind(IISIZE, R2, C2);
   
                    ROWS{c_num} = {R1, R2};
                    COLS{c_num} = {C1, C2};
                    INDS{c_num} = {I1, I2};
                    POLS{c_num} = [1 -1];
                    
                    if D
                        rect_vis(I,ROWS{c_num},COLS{c_num}, POLS{c_num});
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
            for r = 1:SCAN_R_STEP:IMAGE_H - h + 1
                for c = 1:SCAN_C_STEP:IMAGE_W - w + 1
                    
                    
                    R1 = [r r r+h r+h];
                    C1 = [c c+w/3 c c+w/3];
                    R2 = [r r r+h r+h];
                    C2 = [c+w/3 c+(2/3)*w c+w/3 c+(2/3)*w];
                    R3 = [r r r+h r+h];
                    C3 = [c+(2/3)*w c+w c+(2/3)*w c+w];
                    
                    I1 = sub2ind(IISIZE, R1, C1);
                    I2 = sub2ind(IISIZE, R2, C2);
                    I3 = sub2ind(IISIZE, R3, C3);
   
                    ROWS{c_num} = {R1, R2, R3};
                    COLS{c_num} = {C1, C2, C3};
                    INDS{c_num} = {I1, I2, I3};
                    POLS{c_num} = [1 -1 1];
                    
                    if D
                        rect_vis(I,ROWS{c_num},COLS{c_num}, POLS{c_num});
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
            for r = 1:SCAN_R_STEP:IMAGE_H - h + 1
                for c = 1:SCAN_C_STEP:IMAGE_W - w + 1
                   
                    
                    R1 = [r r r+h/3 r+h/3];
                    C1 = [c c+w c c+w];
                    R2 = [r+h/3 r+h/3 r+(2/3)*h r+(2/3)*h];
                    C2 = [c c+w c c+w];
                    R3 = [r+(2/3)*h r+(2/3)*h r+h r+h];
                    C3 = [c c+w c c+w];
                    
                    I1 = sub2ind(IISIZE, R1, C1);
                    I2 = sub2ind(IISIZE, R2, C2);
                    I3 = sub2ind(IISIZE, R3, C3);
   
                    ROWS{c_num} = {R1, R2, R3};
                    COLS{c_num} = {C1, C2, C3};
                    INDS{c_num} = {I1, I2, I3};
                    POLS{c_num} = [1 -1 1];
                    
                    if D
                        rect_vis(I,ROWS{c_num},COLS{c_num}, POLS{c_num});
                    end
                    
                    c_num = c_num + 1;
                end
            end
        end
    end
    if V; disp(['...horz3 classifiers defined in ' num2str(toc) ' seconds']); tic; end;
end


%==========================================================================
% checkerboard HAAR FEATURE
%==========================================================================

if ismember('checker', SHAPES)
    MIN_W = max(2,quant(U_MIN_W,2));                % min feature width
    MIN_H = max(2,quant(U_MIN_H,2));                % min feature height
    W_STEP = max(2,quant(U_STEP_W,2));              % feature width size step
    H_STEP = max(2,quant(U_STEP_H,2));              % feature height size step
 
    for w = MIN_W:W_STEP:MAX_W
        for h = MIN_H:H_STEP:MAX_H      
            for r = 1:SCAN_R_STEP:IMAGE_H - h + 1
                for c = 1:SCAN_C_STEP:IMAGE_W - w + 1
                   
                   
                    R1 = [r r r+h/2 r+h/2];
                    C1 = [c c+w/2 c c+w/2];
                    R2 = [r r r+h/2 r+h/2];
                    C2 = [c+w/2 c+w c+w/2 c+w];
                    R3 = [r+h/2 r+h/2 r+h r+h];
                    C3 = [c c+w/2 c c+w/2];
                    R4 = [r+h/2 r+h/2 r+h r+h];
                    C4 = [c+w/2 c+w c+w/2 c+w];
                    
                    I1 = sub2ind(IISIZE, R1, C1);
                    I2 = sub2ind(IISIZE, R2, C2);
                    I3 = sub2ind(IISIZE, R3, C3);
                    I4 = sub2ind(IISIZE, R4, C4);
   
                    ROWS{c_num} = {R1, R2, R3, R4};
                    COLS{c_num} = {C1, C2, C3, C4};
                    INDS{c_num} = {I1, I2, I3, I4};
                    POLS{c_num} = [1 -1 -1 1];

                    if D
                        rect_vis(I,ROWS{c_num},COLS{c_num}, POLS{c_num});
                    end
                    
                    c_num = c_num + 1;
                end
            end
        end
    end
    if V; disp(['...checker classifiers defined in ' num2str(toc) ' seconds']); end;
end

% get rid of the unused space from the memory allocation
last = c_num - 1;
ROWS = ROWS(1:last);
COLS = COLS(1:last);
INDS = INDS(1:last);
POLS = POLS(1:last);

disp(['...defined ' num2str(last) ' total features.']);
