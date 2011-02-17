function [ cols,  R,C, area, weight] = generate_cross_features(IMSIZE)


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
                
                
               
                
                R1 = [r r+h];
                C1 = [c c+w];

                 
                R2 = [r+(h/3) r+(2*h/3)];
                C2 = [c c+w];

                R3 = [r r+(h/3)];
                C3 = [c+(w/3) c+(2*w/3)];

                R4 = [r+(2*h/3) r+h];
                C4 = [ c+(w/3) c+(2*w/3)];


                %f = [1 0 1 C1(1)-1 R1(1)-1 C1(2)-1 R1(2)-1]; %disp(f);
%                 f = [2 0 1 C1(1)-1 R1(1)-1 C1(2)-2 R1(2)-2 ...
%                       0 -1 C2(1)-1 R2(1)-1 C2(2)-2 R2(2)-2];
                f = [4 0 1 C1(1)-1 R1(1)-1 C1(2)-2 R1(2)-2 ...
                       0 -1 C2(1)-1 R2(1)-1 C2(2)-2 R2(2)-2 ...
                       0 -1 C3(1)-1 R3(1)-1 C3(2)-2 R3(2)-2 ...
                       0 -1 C4(1)-1 R4(1)-1 C4(2)-2 R4(2)-2];   
%                    disp([R1 C1 R2 C2 R3 C3 R4 C4]);
                %disp([R1 C1 R2 C2]);
%                 B = rectRender(f, IMSIZE);
%                 imagesc(B); colormap gray; axis image;
%                 drawnow;
                %pause;


                % a simple way to create the feature is to subtract out the
                % center twice
                cols{count} = [1 -1];

                R{count} = { [R1(1) R1(2)],[R2(1) R2(2)],[R3(1) R3(2)],[R4(1) R4(2)]};
                C{count} = { [C1(1) C1(2)],[C2(1) C2(2)],[C3(1) C3(2)],[C4(1) C4(2)]};

                area{count} = [(R1(2) - R1(1) )*(C1(2) - C1(1) ), ...
                               (R2(2) - R2(1) )*(C2(2) - C2(1) ), ...
                               (R3(2) - R3(1) )*(C3(2) - C3(1) ), ...
                               (R4(2) - R4(1) )*(C4(2) - C4(1) )];

                %a = sum(area{count});
                whitearea = area{count}(1);
                blackarea = area{count}(2:4);
                totalblackarea = sum(area{count}(2:4));

               %  keyboard;
                
                whiteweight = 1;
                blackweight = -2 * blackarea *(whitearea/(totalblackarea^2));

                weight{count} = [whiteweight, blackweight ];

                %disp([area{count} weight{count}]);

               

                count = count + 1;
                %keyboard;

                check = [R1 R2 C1 C2 R3 C3 R4 C4];
                if find(check-2 > 23)
                    disp('problem!');
                    keyboard;
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


disp(['   defined ' num2str(count) ' total cross features.']);