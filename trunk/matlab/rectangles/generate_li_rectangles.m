function [RECTS, COLS] = generate_li_rectangles(N, IMSIZE, SYMM)

IISIZE = IMSIZE + [1 1];
RECTS = cell(N,1);
COLS = cell(N,1);
i = 0;



while i < N


    

    % record of previously placed rectangle(s)
    Q = zeros(IMSIZE);

    
  
    % sample the rank
    RANK = randint([2 4],1);
    rects = cell(1, RANK);
    cols = zeros(1,RANK);
    
    % sample the first rectangle
    r = randint([1 24],1);
    c = randint([1 24],1);
    WMAX = min(floor(IMSIZE(2)/RANK), IISIZE(2) - c );
    HMAX = min(floor(IMSIZE(1)/RANK), IISIZE(1) - r );
    %w = randsample(1:WMAX,1,true,1./(1:WMAX));  if c+w > IISIZE(2); w = WMAX; end;
    %h = randsample(1:HMAX,1,true,1./(1:HMAX));  if r+h > IISIZE(1); h = HMAX; end;
    w = randsample(1:WMAX,1);  if c+w > IISIZE(2); w = WMAX; end;
    h = randsample(1:HMAX,1);  if r+h > IISIZE(1); h = HMAX; end;
    Q(r:r+h-1,c:c+w-1) = 1;
    R = [r r r+h r+h];
   	C = [c c+w c c+w];
    rects{1} = sub2ind(IISIZE, R, C);
    polset = [-1 1];
    cols(1) = randsample(polset,1);
    
    %figure(123); imagesc(Q,[0 4]);
    %pause;
    
    attempts = 0; j = 0;
    
    % sample the rest of the rectangles
    while (j < RANK-1) && (attempts < 50)
    
        %[valid, rect] = valid_rectangles(Q, [w h], SYMM, RANK);
        [valid, rect] = random_rectangle(Q, [w h], SYMM, RANK);
    
        if valid
        	j = j + 1;
            %ind = randsample(size(rect,1),1);
            %rect = rect(ind,:);
            rects{j+1} = rect; 
            [R C] = ind2sub(IMSIZE+[1 1], rect);
            Q(R(1):R(4)-1,C(1):C(4)-1) = j+1;
            if RANK == 2
                cols(j+1) = -1*cols(1);
            end
            if RANK == 3
                if abs(sum(cols)) < 2
                    cols(j+1) = randsample(polset,1);
                else
                    cols(j+1) = -1*cols(1);
                end
            end
            if RANK == 4
                if sum(cols) >= 1
                    cols(j+1) = -1;
                else
                    cols(j+1) = 1;
                end
            end
        else
            %disp('generated a bad one');
            %break;
        end
        attempts = attempts + 1;
    end

    % if we have a valid rectangle, add it
    if valid
        i = i + 1;
        RECTS{i} = rects;
        COLS{i} = cols;
        
        %figure(123);
        %imagesc(Q, [0 4]);
        %figure(3343);
        %rect_vis_ind(zeros(IMSIZE), rects, cols, 1);
        %pause;
    else
        %disp('could not generate a valid rectangle');
        %keyboard;
    end
end







% function [valid, rects] = valid_rectangles(Q, wh, SYMM, RANK)
% IISIZE = size(Q)+[1 1];
% valid = 0; count = 0; BIG = 700; rects = zeros(BIG, 4);
% 
% 
% 
% if SYMM
% 
%     if rand(1) <= .5
%         h = wh(2)-1;
%         w = wh(1)-1;
%     else
%         h = wh(1)-1;
%         w = wh(2)-1;
%     end
% else
%     WMAX = min(floor(IISIZE(2)/RANK));
%     HMAX = min(floor(IISIZE(1)/RANK));
%     w = randsample(1:WMAX,1,true,1./(1:WMAX));
%     h = randsample(1:HMAX,1,true,1./(1:HMAX));
%     %w = randsample(1:WMAX,1);
%     %h = randsample(1:HMAX,1);
% end
% 
% for r1 = 2:IISIZE(1)-h
%     for c1 = 2:IISIZE(2)-w
%         r2 = r1 + h;
%         c2 = c1 + w;
% 
%         if sum(Q(r1-1:r2-1,c1-1:c2-1)) == 0
%             count = count + 1;
%             R = [r1-1 r1-1 r2 r2];
%             C = [c1-1 c2 c1-1 c2];
%             rects(count,:) = sub2ind(IISIZE, R, C);
%             valid = 1;
%         end
%     end
% end
% 
% rects = rects(1:count,:);



function [valid, rect] = random_rectangle(Q, wh, SYMM, RANK)
IISIZE = size(Q)+[1 1]; rect = [];
valid = 0; 



if SYMM
    if rand(1) <= .5
        h = wh(2)-1;
        w = wh(1)-1;
    else
        h = wh(1)-1;
        w = wh(2)-1;
    end
else
    WMAX = min(floor(IISIZE(2)/RANK));
    HMAX = min(floor(IISIZE(1)/RANK));
    w = randsample(1:WMAX,1,true,1./(1:WMAX));
    h = randsample(1:HMAX,1,true,1./(1:HMAX));
    %w = randsample(1:WMAX,1);
    %h = randsample(1:HMAX,1);
end

rmax = IISIZE(1)-h;
cmax = IISIZE(2)-w;
rmin = 2;
cmin = 2;

r1 = randint([rmin rmax],1);
c1 = randint([cmin,cmax],1);
r2 = r1 + h;
c2 = c1 + w;

if sum(Q(r1-1:r2-1,c1-1:c2-1)) == 0
    R = [r1-1 r1-1 r2 r2];
    C = [c1-1 c2 c1-1 c2];
    rect = sub2ind(IISIZE, R, C);
    valid = 1;
end
    
  


