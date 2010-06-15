function [RECTS, COLS] = generate_ramirez_rectangles(N, IMSIZE)

IISIZE = IMSIZE + [1 1];
RECTS = cell(N,1);
COLS = cell(N,1);
i = 0;



while i < N


   
    % sample the feature type
    TYPE = randint([1 6],1);
    
    switch TYPE
        
        case 1 
            rects = cell(1, 2);
            cols = zeros(1, 2);
            WMIN = 3;  WMAX = IISIZE(2);
            HMIN = 2;  HMAX = IISIZE(1);
            w = randsample(WMIN:WMAX, 1, true, 1./(WMIN:WMAX));
            h = randsample(HMIN:HMAX, 1, true, 1./(HMIN:HMAX));
            r = randint([1 HMAX-h],1);
            c = randint([1 WMAX-w],1);
            
            div = randint([c+1 c+w-2],1);
            R1 = [r r r+h-1 r+h-1];
            R2 = [r r r+h-1 r+h-1];
            C1 = [c div c div];
            C2 = [div c+w-1 div c+w-1];
            rects{1} = sub2ind(IISIZE, R1, C1);
            rects{2} = sub2ind(IISIZE, R2, C2);
            cols(1) = 1;
            cols(2) = -1;
            i = i + 1;
            RECTS{i} = rects;
            COLS{i} = cols;
        case 2
            rects = cell(1, 2);
            cols = zeros(1, 2);
            WMIN = 2;  WMAX = IISIZE(2);
            HMIN = 3;  HMAX = IISIZE(1);
            w = randsample(WMIN:WMAX, 1, true, 1./(WMIN:WMAX));
            h = randsample(HMIN:HMAX, 1, true, 1./(HMIN:HMAX));
            r = randint([1 HMAX-h],1);
            c = randint([1 WMAX-w],1);
            
            div = randint([r+1 r+h-2],1);
            R1 = [r r div div];
            R2 = [div div r+h-1 r+h-1];
            C1 = [c c+w-1 c c+w-1];
            C2 = [c c+w-1 c c+w-1];
            rects{1} = sub2ind(IISIZE, R1, C1);
            rects{2} = sub2ind(IISIZE, R2, C2);
            cols(1) = 1;
            cols(2) = -1;
            i = i + 1;
            RECTS{i} = rects;
            COLS{i} = cols;
            
        case 3
            rects = cell(1, 3);
            cols = zeros(1, 3);
            WMIN = 4;  WMAX = IISIZE(2);
            HMIN = 2;  HMAX = IISIZE(1);
            w = randsample(WMIN:WMAX, 1, true, 1./(WMIN:WMAX));
            h = randsample(HMIN:HMAX, 1, true, 1./(HMIN:HMAX));
            r = randint([1 HMAX-h],1);
            c = randint([1 WMAX-w],1);
            
            div1 = randint([c+1 c+w-3],1);
            div2 = randint([div1+1 c+w-2],1);
            R1 = [r r r+h-1 r+h-1];
            R2 = [r r r+h-1 r+h-1];
            R3 = [r r r+h-1 r+h-1];
            C1 = [c div1 c div1];
            C2 = [div1 div2 div1 div2];
            C3 = [div2 c+w-1 div2 c+w-1];
            rects{1} = sub2ind(IISIZE, R1, C1);
            rects{2} = sub2ind(IISIZE, R2, C2);
            rects{3} = sub2ind(IISIZE, R3, C3);
            cols(1) = 1;
            cols(2) = -1;
            cols(3) = 1;
            i = i + 1;
            RECTS{i} = rects;
            COLS{i} = cols;
        case 4
            rects = cell(1, 1);
            cols = zeros(1, 1);
            WMIN = 4;  WMAX = IISIZE(2);
            HMIN = 4;  HMAX = IISIZE(1);
            w = randsample(WMIN:3:WMAX, 1, true, 1./(WMIN:3:WMAX));
            h = randsample(HMIN:3:HMAX, 1, true, 1./(HMIN:3:HMAX));
            r = randint([1 HMAX-h],1);
            c = randint([1 WMAX-w],1);
            
            w2 = ((w-1)/3 )+1;
            h2 = ((h-1)/3 )+1;
            r2 = r+h2-1;
            c2 = c+w2-1;
            

            R1 = [r r r+h-1 r+h-1];
            C1 = [c c+w-1 c c+w-1];
            R2 = [r2 r2 r2+h2-1 r2+h2-1];
            C2 = [c2 c2+w2-1 c2 c2+w2-1];
            R3 = [r2 r2 r2+h2-1 r2+h2-1];
            C3 = [c2 c2+w2-1 c2 c2+w2-1];
            rects{1} = sub2ind(IISIZE, R1, C1);
            rects{2} = sub2ind(IISIZE, R2, C2);
            rects{3} = sub2ind(IISIZE, R3, C3);
            cols(1) = 1;
            cols(2) = -1;
            cols(3) = -1;
            i = i + 1;
            RECTS{i} = rects;
            COLS{i} = cols;
        case 5
            rects = cell(1, 3);
            cols = zeros(1, 3);
            WMIN = 2;  WMAX = IISIZE(2);
            HMIN = 4;  HMAX = IISIZE(1);
            w = randsample(WMIN:WMAX, 1, true, 1./(WMIN:WMAX));
            h = randsample(HMIN:HMAX, 1, true, 1./(HMIN:HMAX));
            r = randint([1 HMAX-h],1);
            c = randint([1 WMAX-w],1);
            
            div1 = randint([r+1 r+h-3],1);
            div2 = randint([div1+1 r+h-2],1);
            
            R1 = [r r div1 div1];
            R2 = [div1 div1 div2 div2];
            R3 = [div2 div2 r+h-1 r+h-1];
            C1 = [c c+w-1 c c+w-1];
            C2 = [c c+w-1 c c+w-1];
            C3 = [c c+w-1 c c+w-1];
            rects{1} = sub2ind(IISIZE, R1, C1);
            rects{2} = sub2ind(IISIZE, R2, C2);
            rects{3} = sub2ind(IISIZE, R3, C3);
            cols(1) = 1;
            cols(2) = -1;
            cols(3) = 1;
            i = i + 1;
            RECTS{i} = rects;
            COLS{i} = cols;
            
        case 6
        
            rects = cell(1, 1);
            cols = zeros(1, 1);
            WMIN = 4;  WMAX = IISIZE(2);
            HMIN = 4;  HMAX = IISIZE(1);
            w = randsample(WMIN:WMAX, 1, true, 1./(WMIN:WMAX));
            h = randsample(HMIN:HMAX, 1, true, 1./(HMIN:HMAX));
            r = randint([1 HMAX-h],1);
            c = randint([1 WMAX-w],1);
            
            ra1 = randint([r+1 r+h-3],1);
            ra2 = randint([ra1+1 r+h-2],1);
            ca1 = randint([c+1 c+w-3],1);
            ca2 = randint([ca1+1 c+w-2],1);

            R1 = [r r r+h-1 r+h-1];
            C1 = [c c+w-1 c c+w-1];
            R2 = [ra1 ra1 ra2 ra2];
            C2 = [ca1 ca2 ca1 ca2];
            R3 = [ra1 ra1 ra2 ra2];
            C3 = [ca1 ca2 ca1 ca2];
            rects{1} = sub2ind(IISIZE, R1, C1);
            rects{2} = sub2ind(IISIZE, R2, C2);
            rects{3} = sub2ind(IISIZE, R3, C3);
            cols(1) = 1;
            cols(2) = -1;
            cols(3) = -1;
            i = i + 1;
            RECTS{i} = rects;
            COLS{i} = cols;
    end
    
end
    