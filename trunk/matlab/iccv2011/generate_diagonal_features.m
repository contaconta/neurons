function [ cols,  R,C, area, weight] = generate_diagonal_features(IMSIZE)


MIN_W = 3;
%MIN_H = 3;
w_list = MIN_W*(1:IMSIZE(2));  w_list = w_list(w_list <= IMSIZE(2));
%h_list = MIN_H*(1:IMSIZE(2));  h_list = h_list(h_list <= IMSIZE(2));
count = 1;

BIG = 150000;
cols = cell(BIG,1);
R = cell(BIG, 1);
C = cell(BIG, 1);
area =  cell(BIG, 1);
weight = cell(BIG, 1);

for w = w_list
    for d = 1:2
    h = w;
        for r = 1:IMSIZE(1) - h + 1
            for c = 1:IMSIZE(2) - w +1
                
               
               
                
                R1 = [r r+h];
                C1 = [c c+w];
                
                R{count} = {R1};
                C{count} = {C1};
                A(1) = h*w;
                cols{count}(1) = 1;

                f = [1 0 1 C1(1)-1 R1(1)-1 C1(2)-2 R1(2)-2];

                for k = 1:h
                    wadd = (w/3)-1;



                    if d == 1
                        cmin = max(c+(k-1)-wadd,c);
                        cmax = min(c+(k-1)+wadd,c + w-1);
                        A(k+1) = cmax-cmin+1;

                        Rk = [r+(k-1) r+(k)];
                        Ck = [cmin cmax+1];
                    else
                        cmin = max(c+(w-k)-wadd,c);
                        cmax = min(c+(w-k)+wadd,c + w-1);
                        A(k+1) = cmax-cmin+1;

                        Rk = [r+(k-1) r+(k)];
                        Ck = [cmin cmax+1];
                    end
                    cols{count}(1) = -1;
                    
                    R{count}{end+1} = Rk;
                    C{count}{end+1} = Ck;

                    f(1) = k +1;
                    f = [f 0 -1 Ck(1)-1 Rk(1)-1 Ck(2)-2 Rk(2)-2]; %#ok<AGROW>
                end


%                 B = rectRender(f, IMSIZE);
%                 imagesc(B); colormap gray; axis image;
%                 drawnow;

                whitearea = A(1);
                blackarea = A(2:end);
                totalblackarea = sum(A(2:end));
                
                whiteweight = 1;
                blackweight = -2 * blackarea *(whitearea/(totalblackarea^2));
                weight{count} = [whiteweight, blackweight ];
                
                count = count + 1;
                
                

           end
        end
    end
%keyboard;
end


cols = cols(1:count-1);
R = R(1:count-1); 
C =  C(1:count-1);
area = area(1:count-1);
weight = weight(1:count-1);


disp(['   defined ' num2str(count) ' total cross features.']);