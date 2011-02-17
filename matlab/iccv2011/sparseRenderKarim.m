function [R s] = sparseRenderKarim(f, IMSIZE,pol)

if ~exist('B', 'var')
    B = zeros(IMSIZE);
end

if ~exist('pol', 'var');
    pol = 1;
end


RANK = f(1);
x = zeros(RANK,1);
y = zeros(RANK,1);
w = zeros(RANK,1);
s = zeros(RANK,1);

p = 2;

for i = 1:RANK
    
    s(i) = f(p);
    x(i) = f(p+1);
    y(i) = f(p+2);
    w(i) = pol*f(p+3);
    
    p = p + 4;
end
    
    

R = reconstruction(IMSIZE, x, y, w, s);

imagesc(R,[-max(abs(R(:))) max(abs(R(:)))]); colormap gray;
drawnow;