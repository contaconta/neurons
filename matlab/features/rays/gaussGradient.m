function [gradient, or] = gaussGradient(im, sigma)

%[rows, cols] = size(im);
im = double(im);         % Ensure double

hsize = [6*sigma+1, 6*sigma+1];   % The filter size.

gaussian = fspecial('gaussian',hsize,sigma);
%im = filter2(gaussian,im);        % Smoothed image.
im = imfilter(im,gaussian, 'replicate', 'full');
[rows, cols] = size(im);

h =  [  im(:,2:cols)  zeros(rows,1) ] - [  zeros(rows,1)  im(:,1:cols-1)  ];
v =  [  im(2:rows,:); zeros(1,cols) ] - [  zeros(1,cols); im(1:rows-1,:)  ];
d1 = [  im(2:rows,2:cols) zeros(rows-1,1); zeros(1,cols) ] - ...
                               [ zeros(1,cols); zeros(rows-1,1) im(1:rows-1,1:cols-1)  ];
d2 = [  zeros(1,cols); im(1:rows-1,2:cols) zeros(rows-1,1);  ] - ...
                               [ zeros(rows-1,1) im(2:rows,1:cols-1); zeros(1,cols)   ];

X = h + (d1 + d2)/2.0;
Y = v + (d1 - d2)/2.0;

b = floor(hsize(1)/2);

X = X(b+1:size(X,1)-b, b+1:size(X,2)-b);
Y = Y(b+1:size(Y,1)-b, b+1:size(Y,2)-b);

gradient = sqrt(X.*X + Y.*Y); % Gradient amplitude.

or = atan2(-Y, X);            % Angles -pi to + pi.
neg = or<0;                   % Map angles to 0-pi.
or = or.*~neg + (or+pi).*neg; 
or = or*180/pi;               % Convert to degrees.