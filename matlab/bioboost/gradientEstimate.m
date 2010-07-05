function [GN, OR] = gradientEstimate(I, scale)

%[rows, cols] = size(I);

sigma = scale/3;
hsize = round([6*sigma+1, 6*sigma+1]);   % The filter size.

gaussian = fspecial('gaussian',hsize,sigma);
I = imfilter(I, gaussian, 'symmetric');        % Smoothed image.

% horizontal difference [-1 0 1]
%h =  [  I(:,2:cols)  zeros(rows,1) ] - [  zeros(rows,1)  I(:,1:cols-1)  ];
%h = [ I(:,2:cols) I(:,cols) ] - [ I(:,1) I(:,1:cols-1) ];
diffc = [-1 0 1];
h = imfilter(double(I), diffc, 'symmetric');
 
% vertical difference [-1; 0; 1]
%v =  [  I(2:rows,:); zeros(1,cols) ] - [  zeros(1,cols); I(1:rows-1,:)  ];
%v =  [  I(2:rows,:); I(rows,:) ] - [  I(1,:); I(1:rows-1,:)  ];
diffr = [-1; 0; 1];
v = imfilter(double(I), diffr, 'symmetric');

%diagonal difference 1 
%d1 = [  I(2:rows,2:cols) zeros(rows-1,1); zeros(1,cols) ] - ...
%                               [ zeros(1,cols); zeros(rows-1,1) I(1:rows-1,1:cols-1)  ];
d1h = [-1 0 0; 0 0 0; 0 0 1];
d1 = imfilter(double(I), d1h, 'symmetric');

% diagonal difference 2                           
%d2 = [  zeros(1,cols); I(1:rows-1,2:cols) zeros(rows-1,1);  ] - ...
%                               [ zeros(rows-1,1) I(2:rows,1:cols-1); zeros(1,cols)   ];
d2h = [0 0 1; 0 0 0; -1 0 0];
d2 = imfilter(double(I), d2h, 'symmetric');

X = h + (d1 + d2)/2.0;
Y = v + (d1 - d2)/2.0;

GN = sqrt(X.*X + Y.*Y); % Gradient amplitude.

OR = atan2(-Y, X);            % Angles -pi to + pi.
neg = OR<0;                   % Map angles to 0-pi.
OR = OR.*~neg + (OR+pi).*neg; 
OR = OR*180/pi;               % Convert to degrees.