function [R s] = sparseRenderKarim(f, IMSIZE,pol,xc,yc)

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
    
    w(i) = pol*f(p);
    s(i) = f(p+1);
    x(i) = f(p+2);
    y(i) = f(p+3);
    
    %[w(i) s(i) x(i) y(i)]
    
    p = p + 4;
end
    
    
[x y w s]

R = reconstruction(IMSIZE, x, y, w, s);

imagesc(R,[-max(abs(R(:))) max(abs(R(:)))]); colormap gray; hold on;
plot(x(w > 0)+1, y(w > 0)+1, 'rs');
        plot(x(w < 0)+1, y(w < 0)+1, 'g.'); 
        plot(xc+1,yc+1, 'mo'); hold off;
drawnow;



%keyboard;