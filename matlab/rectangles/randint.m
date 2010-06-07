function r = randint(a, n)
max = a(2)+1;
min = a(1);
r = floor( (max-min)*rand(1,n)) + min; 