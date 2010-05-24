function [ROWS, COLS, INDS, POLS] = generate_rectangles(N,IMSIZE, RANK)
%
%
%
%
%
%
%

SIGMA = 1;
MAX_ASPECT = 200;

%rects = generate_shape1(10,10,5,1,200);
%figure(1);

ROWS = cell(N,1);
COLS = cell(N,1);
INDS = cell(N,1);
POLS = cell(N,1);

for n = 1:N

        %figure(1);
        rects = generate_shape1(IMSIZE(1),IMSIZE(2),RANK,SIGMA,MAX_ASPECT);

        R = cell(1, size(rects,1));
        C = cell(1, size(rects,1));
        I = cell(1, size(rects,1));
        P = zeros(1, size(rects,1));

        for i = 1:size(rects,1)

            C{i} = [rects(i,1) rects(i,1) rects(i,1)+rects(i,3)+1 rects(i,1)+rects(i,3)+1];
            R{i} = [rects(i,2) rects(i,2)+rects(i,4)+1 rects(i,2) rects(i,2)+rects(i,4)+1];
            I{i} = sub2ind(IMSIZE+[1 1], R{i}, C{i});
            if rects(i,5) == 1
                P(i) = 1;
            else
                P(i) = -1;
            end

        end
        
        %figure(2);
        %rect_vis_ind(zeros(IMSIZE), I, P);
        ROWS{n} = R;
        COLS{n} = C;
        INDS{n} = I;
        POLS{n} = P;
        %keyboard;
        %rects
        
        %pause;
        
end
        
%figure(2);

%rect_vis_ind(zeros(IMSIZE), INDS, POLS);

%rects

%keyboard;