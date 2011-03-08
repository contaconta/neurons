function R = reconstruction_coarse(IMSIZE, x,y,w,sigma,coarseflag)

if ~exist('coarseflag', 'var')
    coarseflag = 1;
end

badinds = find (x < 0);
badinds = [badinds; find( x >= IMSIZE(2))];
badinds = [badinds; find( y < 0)];
badinds = [badinds; find( y >= IMSIZE(1))];

%x = x( x > 0);
%y = y( y > 0);
%x = x( x <= IMSIZE(2));
%y = y( y <= IMSIZE(1));

x(badinds) = [];
y(badinds) = [];
w(badinds) = [];
sigma(badinds) = [];

x = x + 1;
y = y + 1;

R = zeros(IMSIZE);

for j = 1:numel(x)
    
    T = zeros(IMSIZE);
    T(y(j),x(j)) = 1;
    if coarseflag == 1
        T = imgaussian2(T, sigma(j));
    else
        T = imgaussian(T, sigma(j));
    end
               R = R + w(j).*T;
    
    %keyboard;
    
end