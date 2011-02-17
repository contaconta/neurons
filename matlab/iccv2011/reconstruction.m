function R = reconstruction(IMSIZE, x, y, w, sigma) 

[X, Y] = meshgrid(1:IMSIZE(1), 1:IMSIZE(2));


x = x + 1;
y = y + 1;

R = zeros(IMSIZE);

 

for j = 1:numel(sigma)

    R = R + w(j) * gaussian(X, Y, [x(j), y(j)], sigma(j));

    %keyboard;

end









%keyboard;