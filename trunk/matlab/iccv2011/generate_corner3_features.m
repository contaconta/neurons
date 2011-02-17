function [ cols,  R,C, area, weight] = generate_corner3_features(IMSIZE)


MIN_W = 3;
MIN_H = 3;
w_list = MIN_W*(1:IMSIZE(2));  w_list = w_list(w_list <= IMSIZE(2));
h_list = MIN_H*(1:IMSIZE(2));  h_list = h_list(h_list <= IMSIZE(2));
count = 1;

BIG = 150000;
cols = cell(BIG,1);
R = cell(BIG, 1);
C = cell(BIG, 1);
area =  cell(BIG, 1);
weight = cell(BIG, 1);

for w = w_list
    for h = h_list
        for r = 1:IMSIZE(1) - h + 1
            for c = 1:IMSIZE(2) - w +1
                
                
                for k = 1:4
                
                    R1 = [r r+h];
                    C1 = [c c+w];
                    
                    switch k
                        case 1
                            R2 = [r r+(2*h/3)];
                            C2 = [c c+(2*w/3)];
                        case 2
                            R2 = [r r+(2*h/3)];
                            C2 = [c+(w/3) c+w];
                        case 3
                            R2 = [r+(h/3) r+h];
                            C2 = [c c+(2*w/3)];
                        case 4
                            R2 = [r+(h/3) r+h];
                            C2 = [c+(w/3) c+w];
                    end
                    
                    %f = [1 0 1 C1(1)-1 R1(1)-1 C1(2)-1 R1(2)-1]; %disp(f);
%                     f = [2 0 1 C1(1)-1 R1(1)-1 C1(2)-2 R1(2)-2 ...
%                            0 -1 C2(1)-1 R2(1)-1 C2(2)-2 R2(2)-2];
%                     B = rectRender(f, IMSIZE);
%                     imagesc(B); colormap gray; axis image;
%                     drawnow;
                    %pause;


                    % a simple way to create the feature is to subtract out the
                    % center twice
                    cols{count} = [1 -1];
                    
                    R{count} = { [R1(1) R1(2)], [R2(1) R2(2)]};
                    C{count} = { [C1(1) C1(2)], [C2(1) C2(2)]};

                    area{count} = [(R1(2) - R1(1) )*(C1(2) - C1(1) ), (R2(2) - R2(1) )*(C2(2) - C2(1) )];

                    a = sum(area{count});
                    whitearea = area{count}(1);
                    blackarea = area{count}(2);

                    whiteweight = 1;
                    blackweight = -2 * (whitearea/blackarea);

                    weight{count} = [whiteweight, blackweight ];

                    %disp([area{count} weight{count}]);
                    
                    
                    count = count + 1;
                    %keyboard;

                    check = [R1 R2 C1 C2];
                    if find(check-2 > 23)
                        disp('problem!');
                        keyboard;
                    end
                end
                
            end
        end
    end
end


cols = cols(1:count-1);
R = R(1:count-1); 
C =  C(1:count-1);
area = area(1:count-1);
weight = weight(1:count-1);


disp(['   defined ' num2str(count) ' total 3-corner features.']);