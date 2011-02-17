function [X Y W S] = randomGridDiracSampling(f, IMSIZE)

%BLANK = zeros(IMSIZE);

%BW = rectRender(f, IMSIZE, BLANK);
%clf; imagesc(BW); colormap gray; hold on;

RANK = f(1);

p = 2;

X = [];  Y = [];  W = [];  S = [];

for i = 1:RANK
    
 
    w = f(p+1);
    x0 = f(p+2)+1;
    y0 = f(p+3)+1;
    x1 = f(p+4)+1;
    y1 = f(p+5)+1;

    pol = sign(w);
    
    
    h = y1-y0+1;
    w = x1-x0+1;
    
    area = w*h;
    
    l = zeros(area, 2);
    
    count = 1;
    for x = x0:x1
        for y = y0:y1
            l(count,:) = [x y];
            count = count + 1;
        end
    end
    
    inds = randsample(size(l,1), min(size(l,1), round(sqrt(area))));
    
    if pol == 1
        col = 'rs';
    else
        col = 'gs';
    end
    
    %plot(l(inds,1), l(inds,2), col);
    
    
    p = p + 6;
    
    x = l(inds,1)-1;
    y = l(inds,2)-1;
    w = ones(numel(x),1) / numel(x);
    s = zeros(numel(x),1);
    
    X = [X; x];
    Y = [Y; y];
    W = [W; w];
    S = [S; s];
    
end

hold off;


