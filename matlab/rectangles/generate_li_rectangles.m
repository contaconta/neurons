function [RECTS, COLS] = generate_li_rectangles(N, IMSIZE, SYMM)

IISIZE = IMSIZE + [1 1];
RECTS = cell(N,1);
COLS = cell(N,1);
i = 0;



while i < N
    % record of previously placed rectangle(s)
    Q = zeros(IMSIZE);
    
    % sample the rank
    RANK = randint([1 4],1);
    rects = {};
    cols = [];
    
    %% RANK 2
    if RANK == 2
        % generate the first rectangle
        r = randint([1 IMSIZE(1)],1);
        c = randint([1 IMSIZE(2)],1);
        WMAX = min(floor(IMSIZE(2)/RANK), IISIZE(2) - c );
        HMAX = min(floor(IMSIZE(1)/RANK), IISIZE(1) - r );
        w = randsample(1:WMAX,1);  if c+w > IISIZE(2); w = WMAX; end;
        h = randsample(1:HMAX,1);  if r+h > IISIZE(1); h = HMAX; end;
        Q(r:r+h-1,c:c+w-1) = 1;
        R = [r r r+h r+h];
        C = [c c+w c c+w];
        rects{1} = sub2ind(IISIZE, R, C);
        polset = [-1 1];
        cols(1) = randsample(polset,1);
        
        j = 0; attempts = 0;
        while (j < 1) && (attempts < 50)
            [valid, rect] = random_rectangle(Q, [w h], SYMM, RANK);
            if valid
                rects{2} = rect; 
                cols(2) = -1*cols(1);
                i = i + 1; j = j + 1;
                RECTS{i} = rects;
                COLS{i} = cols;
            end
            attempts = attempts + 1;
        end
    end
    
    %% RANK 3
    if RANK == 3
        % generate the first rectangle
        r = randint([1 IMSIZE(1)],1);
        c = randint([1 IMSIZE(2)],1);
        WMAX = min(floor(IMSIZE(2)/RANK), IISIZE(2) - c );
        HMAX = min(floor(IMSIZE(1)/RANK), IISIZE(1) - r );
        w = randsample(1:WMAX,1);  if c+w > IISIZE(2); w = WMAX; end;
        h = randsample(1:HMAX,1);  if r+h > IISIZE(1); h = HMAX; end;
        Q(r:r+h-1,c:c+w-1) = 1;
        R = [r r r+h r+h];
        C = [c c+w c c+w];
        rects = {}; rects{1} = sub2ind(IISIZE, R, C);
        polset = [-1 1];
        cols(1) = randsample(polset,1);
        
        % generate the 2nd rectangle
        attempts = 0;
        while (length(rects) < 2) && (attempts < 50)
            [valid, rect] = random_rectangle(Q, [w h], SYMM, RANK, r); % fix r to be the same
            if valid
                rects{2} = rect; 
                cols(2) = -1*cols(1);
                [R C] = ind2sub(IMSIZE+[1 1], rect);
                Q(R(1):R(4)-1,C(1):C(4)-1) = 2;
            end
            attempts = attempts + 1;
        end
        
        % generate the 3rd rectangle
        while (length(rects) < 3) && (attempts < 50)
            [valid, rect] = random_rectangle(Q, [w h], SYMM, RANK); 
            if valid
                rects{3} = rect; 
                cols(3) = randsample(polset,1);
                i = i + 1;
                RECTS{i} = rects;
                COLS{i} = cols;
                %keyboard;
            end
            attempts = attempts + 1;
        end
    end
    
    %% RANK 4
    if RANK == 4
        % generate the first rectangle
        r = randint([1 IMSIZE(1)],1);
        c = randint([1 IMSIZE(2)],1);
        WMAX = min(floor(IMSIZE(2)/RANK), IISIZE(2) - c );
        HMAX = min(floor(IMSIZE(1)/RANK), IISIZE(1) - r );
        w = randsample(1:WMAX,1);  if c+w > IISIZE(2); w = WMAX; end;
        h = randsample(1:HMAX,1);  if r+h > IISIZE(1); h = HMAX; end;
        Q(r:r+h-1,c:c+w-1) = 1;
        R = [r r r+h r+h];
        C = [c c+w c c+w];
        rects = {}; rects{1} = sub2ind(IISIZE, R, C);
        polset = [-1 1];
        cols(1) = randsample(polset,1);
        
        % generate the 2nd rectangle
        attempts = 0;
        while (length(rects) < 2) && (attempts < 50)
            [valid, rect] = random_rectangle(Q, [w h], SYMM, RANK, r); % fix r to be the same
            if valid
                rects{2} = rect; 
                cols(2) = -1*cols(1);
                [R C] = ind2sub(IMSIZE+[1 1], rect);
                Q(R(1):R(4)-1,C(1):C(4)-1) = 2;
            end
            attempts = attempts + 1;
        end
        
        % place the 3rd and 4th rectangles to be similar to 1 and 2
        while (length(rects) < 4) && (attempts < 50)
            [valid, rect34] = place_rects(Q, rects);
            if valid
                rects{3} = rect34{1};
                rects{4} = rect34{2};
                cols(3) = randsample(polset,1);
                cols(4) = -1*cols(3);
                i = i + 1;
                RECTS{i} = rects;
                COLS{i} = cols;
            end
            attempts = attempts + 1;
        end
    end
    
    
    
end


function [valid, rect34] = place_rects(Q, rects)
valid = 0; rect34 = {};

IISIZE = size(Q)+[1 1];

[R1 C1] = ind2sub(IISIZE, rects{1});
[R2 C2] = ind2sub(IISIZE, rects{2});

rs_min = min([R1,R2]) - 2; 
rs_max = IISIZE(1) - max([R1 R2]) - 1;
cs_min = min([C1,C2]) - 2;
cs_max = IISIZE(2) - max([C1 C2]) - 1;

rs = randint([-rs_min rs_max],1);
cs = randint([-cs_min cs_max],1);

R3 = R1 + rs;
R4 = R2 + rs;
C3 = C1 + cs;
C4 = C2 + cs;

if (R3(1) > 1) && (R4(1) > 1) && (R3(4) <= IISIZE(1)) && (R4(4) <= IISIZE(1)) && (C3(1) > 1) && (C4(1) > 1) && (C3(4) <= IISIZE(2)) && (C4(4) <= IISIZE(2)) 
    if (sum(sum(Q(R3(1)-1:R3(4)-1,C3(1)-1:C3(4)-1))) == 0) && (sum(sum(Q(R4(1)-1:R4(4)-1,C4(1)-1:C4(4)-1))) == 0)
        rect34{1} = sub2ind(IISIZE, R3, C3);
        rect34{2} = sub2ind(IISIZE, R4, C4);
        valid = 1;
    end
end





function [valid, rect] = random_rectangle(Q, wh, SYMM, RANK, rfix)
IISIZE = size(Q)+[1 1]; rect = [];
valid = 0; 

if nargin < 5
    rfix = 0;
else
    rfix = rfix + 1;
end

if SYMM

        h = wh(2)-1;
        w = wh(1)-1;

else
    WMAX = min(floor(IISIZE(2)/RANK));
    HMAX = min(floor(IISIZE(1)/RANK));
    w = randsample(1:WMAX,1,true,1./(1:WMAX));
    h = randsample(1:HMAX,1,true,1./(1:HMAX));
end

rmax = IISIZE(1)-h;
cmax = IISIZE(2)-w;
rmin = 2;
cmin = 2;

if rfix == 0
    r1 = randint([rmin rmax],1);
else
    r1 = rfix;
end
c1 = randint([cmin,cmax],1);
r2 = r1 + h;
c2 = c1 + w;

if sum(sum(Q(r1-1:r2-1,c1-1:c2-1))) == 0
    R = [r1-1 r1-1 r2 r2];
    C = [c1-1 c2 c1-1 c2];
    rect = sub2ind(IISIZE, R, C);
    valid = 1;
end

