function r = randint(a, n)

if nargin < 2
    n = 1;
end

max = a(2)+1;
min = a(1);
r = floor( (max-min)*rand(1,n)) + min; 