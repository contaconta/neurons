function B = rectRender(f, IMSIZE, B)

if ~exist('B', 'var')
    B = zeros(IMSIZE);
end


RANK = f(1);

p = 2;


for i = 1:RANK
        
    
    TILT = f(p);
    
    if ~TILT
        w = f(p+1);
        x0 = f(p+2)+1;
        y0 = f(p+3)+1;
        x1 = f(p+4)+1;
        y1 = f(p+5)+1;
        
        pol = sign(w);
        
        B(y0:y1,x0:x1) = pol;
        
        valid = boundary_check(x0,x1,y0,y1,IMSIZE);
        if ~valid
            error('element is out of image boundaries!');
        end
        
        p = p + 6;
        
    else
        [pat, p] = drawpatch(f,p);
        %[B p] = fill45(B, f, p);
        %disp('titled feature not handled yet');
    end    
end

%keyboard;


function [pat, p]  = drawpatch(f,p)

w = f(p+1);
pol = sign(w);

A = [ f(p+2)+1 f(p+3)+1] - [.5 .5];
B = [ f(p+4)+1 f(p+5)+1] - [.5 .5];
C = [ f(p+6)+1 f(p+7)+1] - [.5 .5];
D = [ f(p+8)+1 f(p+9)+1] - [.5 .5];

if pol == 1
    color = [1 1 1];
else
    color = [0 0 0];
end

X = [A(1) B(1) D(1) C(1)];
Y = [A(2) B(2) D(2) C(2)];

figure(1);
pat = patch(X,Y,color);
set(pat, 'Parent', gca);
%keyboard;

p = p +10;


function TF = boundary_check(x0,x1,y0,y1,IMSIZE)

TF = 1;

if x0 < 1
    TF = 0;
end

if x1 > IMSIZE(2)
    TF = 0;
end

if y0 < 1
    TF = 0;
end

if y1 > IMSIZE(1)
    TF = 0;
end






% if find( (X - .5) < 0)
%     disp('too low X');
%     keyboard;
% end
% 
% 
% if find( (X - .5) > 23)
%     disp('too high X');
%     keyboard;
% end
% 
% 
% if find( (Y - .5) < 0)
%     disp('too low Y');
%     keyboard;
% end
% 
% 
% if find( (Y - .5) > 23)
%     disp('too high Y');
%     keyboard;
% end
% pat = [];






% function [I p] = fill45(I, f, p)
% 
% %keyboard;
% 
% w = f(p+1);
% pol = sign(w);
% 
% A = [f(p+3)+1 f(p+2)+1];
% B = [f(p+5)+1 f(p+4)+1];
% C = [f(p+7)+1 f(p+6)+1];
% D = [f(p+9)+1 f(p+8)+1];
% 
% p = p +10;
% 
% BW = false(size(I));
% 
% BW = drawline(BW, A, C, -sign(A-C));
% BW = drawline(BW, A, B, -sign(A-B));
% BW = drawline(BW, B, D, -sign(B-D));
% BW = drawline(BW, C, D, -sign(C-D));
% %loc = sub2ind(size(BW),A(1)+1, A(2));
% BW = imfill(BW, [A(1)+1 A(2)], 4);
% 
% %BW = double(BW);
% I(BW == 1) = pol;
% %keyboard;
% 
% 
% function BW = drawline(BW,start,stop,vec)
% 
% r = start(1); c = start(2);
% pr = sign(stop(1) - start(1));  % + indicated increasing from start to stop
% pc = sign(stop(2) - start(2));  % + indicates increasing from start to stop
% 
% if start(1) == stop(1)
%     BW(r,c) = 1;
% else
% 
%     while (pr*r <= pr*stop(1)) && (r > 0) && (r <= size(BW,1)) && (pc*c <= pc*stop(2)) && (c > 0) && (c <= size(BW,2))
%         BW(r,c) = 1;
%         r = r+vec(1);
%         c = c+vec(2);
%     end
% end
