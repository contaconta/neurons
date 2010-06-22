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

ROWS = cell(N,1);
COLS = cell(N,1);
INDS = cell(N,1);
POLS = cell(N,1);

%RANKBAR = zeros(1, RANK);

for n = 1:N

        rects = []; redo = 0;
        while isempty(rects)
            
            % first, sample the subwindow size
            w = randint(24);
            h = randint(24);

            % next sample an offset for the subwindow
            ro = randint([0 IMSIZE(2) - w],1);
            co = randint([0 IMSIZE(1) - h],1);

            % sample the rank
            rank = randint([2 RANK]);
            
            % generate a shape with sampled parameters
            rects = generate_shape1(w,h,rank,SIGMA,MAX_ASPECT);
            %if redo == 1; disp('needed to regenerate'); end;
            %redo  = 1;
        end
        
        % apply the sub-window offset
        rects(:,1) = rects(:,1) + ro;
        rects(:,2) = rects(:,2) + co;

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
        
%         figure(3);
%         rect_vis_ind(zeros(IMSIZE), I, P, 1);
        ROWS{n} = R;
        COLS{n} = C;
        INDS{n} = I;
        POLS{n} = P;

        
%         RANKBAR(1, size(rects,1)) = RANKBAR(1, size(rects,1))+1;
%         figure(2);
%         bar(RANKBAR);   
end
 