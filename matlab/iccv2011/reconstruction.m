function R = reconstruction(IMSIZE, x, y, w, sigma) 

[X, Y] = meshgrid(1:IMSIZE(1), 1:IMSIZE(2));

x = x + 1;
y = y + 1;

badinds = find (x < 1);
badinds = [badinds; find( x > IMSIZE(2))];
badinds = [badinds; find( y < 1)];
badinds = [badinds; find( y > IMSIZE(1))];

%x = x( x > 0);
%y = y( y > 0);
%x = x( x <= IMSIZE(2));
%y = y( y <= IMSIZE(1));

x(badinds) = [];
y(badinds) = [];
w(badinds) = [];
sigma(badinds) = [];




R = zeros(IMSIZE);

 

for j = 1:numel(sigma)

    R = R + w(j) * gaussian(X, Y, [x(j), y(j)], sigma(j));

    %keyboard;

end









%keyboard;