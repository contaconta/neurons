function  [rects cols areas types] = generate_lienhart_features(IMSIZE, NORM)

if ~ismember(NORM, { 'ANORM','NONORM','DNORM'})
    disp('Error: incorrect NORM specified');
    keyboard;
end


%% generate the extended viola-jones features
[R,C,rect,col] = generate_viola_jones_features(IMSIZE);
rects = rect;
cols = col;
types(1:length(rects),1) = deal({'Lienhart90'});

% generate the '1-2-1' features
[R,C,rect,col] = generate_viola_jones_features_special(IMSIZE, 'shapes', 'horz3', 'vert3');
rects = [rects; rect];
cols = [cols; col];
type(1:length(rect),1) = deal({'Lienhart90'});  types = [types; type];

% generate the surrounding box features
disp('...generating the center surround');
[rect, col, type] = generate_center_surround(IMSIZE);
rects = [rects; rect];
cols = [cols; col];
type(1:length(rect),1) = deal({'Lienhart90'});  types = [types; type];

% compute the areas for rectilinear features as usual
disp('...computing areas for 90 degree features');
areas = compute_areas2(IMSIZE, NORM, rects, cols);



%% generate the special lienhart 45degree features
IISIZE = IMSIZE + [ 1 2];
[rect, col, type] = generate_45_features(IMSIZE);
rects = [rects; rect];
cols = [cols; col];
types = [types; type];

% compute the areas specially
area = cell(size(rect));
disp('...computing areas for 45 degree features');
switch NORM
    case 'ANORM'
        for i = 1:length(rect)
            
            BW = rect_vis_ind45(zeros(IISIZE), rect{i}, col{i},1,0);
            area{i} = [sum(BW(:)==1), sum(BW(:)==-1)];
        end
    case 'DNORM'
        for i = 1:length(rect)
            BW = rect_vis_ind45(zeros(IISIZE), rect{i}, col{i},1,0);
            a = [sum(BW(:)==1), sum(BW(:)==-1)];
            if a(1) == a(2)
                area{i} = [0 0];
            else
                area{i} = a;
            end
        end
    case 'NONORM'
        for i = 1:length(rect)
            area{i} = [0 0];
        end
end
areas = [areas; area];

disp(['...generated ' num2str(length(rects)) ' total Lienhart features.']);










function [rects, cols, types] = generate_center_surround(IMSIZE)

IISIZE = IMSIZE + [ 1 1];
MIN_W = 3;
MIN_H = 3;
w_list = MIN_W*(1:IMSIZE(2));  w_list = w_list(w_list <= IMSIZE(2));
h_list = MIN_H*(1:IMSIZE(2));  h_list = h_list(h_list <= IMSIZE(2));
count = 1;

BIG = 150000;
rects = cell(BIG,1);
cols = cell(BIG,1);
types = cell(BIG, 1);

for w = w_list
    for h = h_list
        for r = 1:IMSIZE(1) - h + 1
            for c = 1:IMSIZE(2) - w + 1
                R1 = [r r r+h r+h];
                C1 = [c c+w c c+w];
                
                R2 = [r+h/3 r+(2/3)*h r+h/3 r+(2/3)*h];
                C2 = [c+w/3 c+(2/3)*w c+w/3 c+(2/3)*w];

                I1 = sub2ind(IISIZE, R1, C1);
                I2 = sub2ind(IISIZE, R2, C2);

                % a simple way to create the feature is to subtract out the
                % center twice
                rects{count} = {I1, I2, I2};
                cols{count} = [1 -1 -1];
                types{count} = 'LienhartInnerBox';
                
                %rect_vis_ind(zeros(24,24),rects{count},cols{count}, 1);
                
                count = count + 1;
                
            end
        end
    end
end

rects = rects(1:count-1);
cols = cols(1:count-1);
types = types(1:count-1);

disp(['...defined ' num2str(count) ' total center-surround Lienhart90 features.']);