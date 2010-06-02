function r = randint(a, n)
max = a(2);
min = a(1);
r = round( (max-min)*rand(1,n)) + min; 