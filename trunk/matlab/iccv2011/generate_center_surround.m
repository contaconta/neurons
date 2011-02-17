
function [rects, cols, types, R,C, area, weight] = generate_center_surround(IMSIZE)

%IISIZE = IMSIZE + [ 1 1];
MIN_W = 3;
MIN_H = 3;
w_list = MIN_W*(1:IMSIZE(2));  w_list = w_list(w_list <= IMSIZE(2));
h_list = MIN_H*(1:IMSIZE(2));  h_list = h_list(h_list <= IMSIZE(2));
count = 1;

BIG = 150000;
rects = cell(BIG,1);
cols = cell(BIG,1);
types = cell(BIG, 1);
R = cell(BIG, 1);
C = cell(BIG, 1);
area =  cell(BIG, 1);
weight = cell(BIG, 1);

for w = w_list
    for h = h_list
        for r = 1:IMSIZE(1) - h + 1
            for c = 1:IMSIZE(2) - w +1 
                R1 = [r r r+h-1 r+h-1];
                C1 = [c c+w-1 c c+w-1];
                
                R2 = [r+h/3 (r+(2/3)*h)-1 r+h/3 (r+(2/3)*h)-1];
                C2 = [c+w/3 (c+(2/3)*w)-1 c+w/3 (c+(2/3)*w)-1];

                R1 = [R1(1) R1(4)] + [0 1];
                R2 = [R2(1) R2(4)] + [0 1];

                C1 = [C1(1) C1(4)] + [0 1];
                C2 = [C2(1) C2(4)] + [0 1];
                
                %I1 = sub2ind(IISIZE, R1, C1);
                %I2 = sub2ind(IISIZE, R2, C2);

                % a simple way to create the feature is to subtract out the
                % center twice
                rects{count} = {[]};
                cols{count} = [1 -1];
                types{count} = 'LienhartInnerBox';
                
                R{count} = { [R1(1) R1(2)], [R2(1) R2(2)]};
                C{count} = { [C1(1) C1(2)], [C2(1) C2(2)]};
                
                area{count} = [(R1(2) - R1(1) )*(C1(2) - C1(1) ), (R2(2) - R2(1) )*(C2(2) - C2(1) )];
                
                a = sum(area{count});
                
                whitearea = area{count}(1);
                blackarea = area{count}(2);
                
                whiteweight = 1;
                blackweight = -2 * (whitearea/blackarea);
                
                weight{count} = [whiteweight, blackweight ];
                
                %rect_vis_ind(zeros(24,24),rects{count},cols{count}, 1);
                
                count = count + 1;
                
                %keyboard;
                
            end
        end
    end
end

rects = rects(1:count-1);
cols = cols(1:count-1);
types = types(1:count-1);
R = R(1:count-1); 
C =  C(1:count-1);
area = area(1:count-1);
weight = weight(1:count-1);


disp(['   defined ' num2str(count) ' total center-surround features.']);