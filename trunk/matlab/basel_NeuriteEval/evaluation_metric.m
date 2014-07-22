function [t s m1 m2] = evaluation_metric(r1,c1,r2,c2,sigma)


if isempty(r1)
    t = [];
    s = [];
    m1 = 0;
    m2 = 0;
    return;
end

if isempty(r2)
    t = ones(size(r1));
    s = ones(size(r1));
    m1 = 1;
    m2 = 1;
    return
end



N = numel(r1);

t = zeros(N,1);
s = zeros(N,1);

for i = 1:N
    
    d = sqrt( (r1(i)-r2).^2 + (c1(i) - c2).^2);
    
    mind(i) = min(d);
    
    
    if mind(i) <= sigma
        t(i) = 0;
    else
        t(i) = 1;
    end
    
    s(i) = 1 - exp(  -(mind(i)^2) / (2*sigma^2));
    
end


[t  mind(:) s];

m1 = 1/N * (sum(t));

m2 = 1/N * sum(s);

[m1 m2];