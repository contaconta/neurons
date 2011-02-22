function [S, box] = GetSizeSubWindow(f)


RANK = f(1);

p = 2;

for i = 1:RANK
        
    
    TILT = f(p);
    
    if ~TILT
        x0 = f(p+2); box.x0 = x0+1;
        y0 = f(p+3); box.y0 = y0+1;
        x1 = f(p+4); box.x1 = x1+1;
        y1 = f(p+5); box.y1 = y1+1;
        
        S = (x1-x0+1)*(y1-y0+1);
        return;
    end
end

