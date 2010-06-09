function [rects, cols, types] = generate_45_features(IMSIZE)

IISIZE = IMSIZE + [ 1 2];
%VEC_SHIFT = (IMSIZE(1)+1)  * (IMSIZE(2)+1); % how far we must shift the linear indexes for the 45 degree integral image
VEC_SHIFT = 25*25;
cnt = 1;

% PRE-ALLOCATE RECTS, etc...
BIG = 150000;
rects = cell(BIG,1);
cols = cell(BIG,1);
%areas = cell(BIG, 1);
types = cell(BIG, 1);


for r = 2:IISIZE(1)-2
    for c = 1:IISIZE(2)-2
        % place point C
        C = [r c];
        
        cdmax = exploreline(IISIZE, C, [-1 1]);
        cdsteps = C(1) - cdmax(1);
        for cd = 2:cdsteps
            D = C + cd*[-1 1];
            
            camax = exploreline(IISIZE, C, [1 1]);
            casteps = camax(1) - C(1);
            for ca = 2:casteps
                A = C + ca*[1 1];
                B = A + cd*[-1 1];
            
                ROW = [A(1) B(1) C(1) D(1)];
                COL = [A(2) B(2) C(2) D(2)];
                
                if (ca > 1) || (cd > 1)
                    if isempty(find(ROW > IISIZE(1),1)) && isempty(find(ROW < 1,1))
                        if isempty(find(COL > IISIZE(2),1)) && isempty(find(COL < 1,1))

                            % WE KNOW WE HAVE A VALID CONTAINER. FIND VALID
                            % PARTITIONS
                            
                            % 2 partition along ca
                            if mod(ca,2) == 0
                                C1 = C; D1 = D;
                                A1 = C1 + (ca/2)*[1 1];
                                B1 = A1 + cd*[-1 1];
                                
                                C2 = A1; D2 = B1;
                                A2 = A; B2 = B;
                                
                                rect1 = ABCDtoRect(A1,B1,C1,D1, VEC_SHIFT, IISIZE);
                                rect2 = ABCDtoRect(A2,B2,C2,D2, VEC_SHIFT, IISIZE);
                                rects{cnt} = {rect1, rect2};
                                cols{cnt} = [1 -1];
                                types{cnt} = 'Lienhart45';

                                %rect_vis_ind45(zeros(IISIZE), rects{cnt}, cols{cnt},1);
                                %BW = rect_vis_ind45(zeros(IISIZE), rects{cnt}, cols{cnt},1);
                                %areas{cnt} = [sum(BW(:)==1), sum(BW(:)==-1)];
                                cnt = cnt + 1;
                            end
                            
                            % 2 partition along cd
                            if mod(cd,2) == 0
                                C1 = C; A1 = A;
                                D1 = C1 + (cd/2)*[-1 1];
                                B1 = A1 + (cd/2)*[-1 1];
                                
                                C2 = D1; D2 = D;
                                A2 = B1; B2 = B;
                                
                                rect1 = ABCDtoRect(A1,B1,C1,D1, VEC_SHIFT, IISIZE);
                                rect2 = ABCDtoRect(A2,B2,C2,D2, VEC_SHIFT, IISIZE);
                                rects{cnt} = {rect1, rect2};
                                cols{cnt} = [1 -1];
                                types{cnt} = 'Lienhart45';

                                %rect_vis_ind45(zeros(IISIZE), rects{cnt}, cols{cnt},1);
                                cnt = cnt + 1;
                            end
                            
                            % 3 partition along ca
                            if mod(ca,3) == 0
                                C1 = C; D1 = D;
                                A1 = C1 + (ca/3)*[1 1];
                                B1 = A1 + cd*[-1 1];
                                
                                C2 = A1; D2 = B1;
                                A2 = C2 + (ca/3)*[1 1];
                                B2 = A2 + cd*[-1 1];
                                
                                C3 = A2; D3 = B2;
                                A3 = A; B3 = B;
                                
                                rect1 = ABCDtoRect(A1,B1,C1,D1, VEC_SHIFT, IISIZE);
                                rect2 = ABCDtoRect(A2,B2,C2,D2, VEC_SHIFT, IISIZE);
                                rect3 = ABCDtoRect(A3,B3,C3,D3, VEC_SHIFT, IISIZE);
                                rects{cnt} = {rect1, rect2, rect3};
                                cols{cnt} = [1 -1 1];
                                types{cnt} = 'Lienhart45';

                                %rect_vis_ind45(zeros(IISIZE), rects{cnt}, cols{cnt},1);
                                cnt = cnt + 1;
                                
                            end
                            
                            % 3 partition along cd
                            if mod(cd,3) == 0
                                C1 = C; A1 = A;
                                D1 = C1 + (cd/3)*[-1 1];
                                B1 = A1 + (cd/3)*[-1 1];
                                
                                C2 = D1; A2 = B1;
                                D2 = C2 + (cd/3)*[-1 1];
                                B2 = A2 + (cd/3)*[-1 1];
                                
                                C3 = D2; D3 = D;
                                A3 = B2; B3 = B;
                                
                                rect1 = ABCDtoRect(A1,B1,C1,D1, VEC_SHIFT, IISIZE);
                                rect2 = ABCDtoRect(A2,B2,C2,D2, VEC_SHIFT, IISIZE);
                                rect3 = ABCDtoRect(A3,B3,C3,D3, VEC_SHIFT, IISIZE);
                                rects{cnt} = {rect1, rect2, rect3};
                                cols{cnt} = [1 -1 1];
                                types{cnt} = 'Lienhart45';

                                %rect_vis_ind45(zeros(IISIZE), rects{cnt}, cols{cnt},1);
                                cnt = cnt + 1;
                                
                            end
                            
                            % the vj-special at 45 along ca
                            if mod(ca,4) == 0
                                C1 = C; D1 = D;
                                A1 = C1 + (ca/4)*[1 1];
                                B1 = A1 + cd*[-1 1];
                                
                                C2 = A1; D2 = B1;
                                A2 = C2 + (ca/2)*[1 1];
                                B2 = A2 + cd*[-1 1];
                                
                                C3 = A2; D3 = B2;
                                A3 = A; B3 = B;
                                
                                rect1 = ABCDtoRect(A1,B1,C1,D1, VEC_SHIFT, IISIZE);
                                rect2 = ABCDtoRect(A2,B2,C2,D2, VEC_SHIFT, IISIZE);
                                rect3 = ABCDtoRect(A3,B3,C3,D3, VEC_SHIFT, IISIZE);
                                rects{cnt} = {rect1, rect2, rect3};
                                cols{cnt} = [1 -1 1];
                                types{cnt} = 'Lienhart45';

                                %rect_vis_ind45(zeros(IISIZE), rects{cnt}, cols{cnt},1);
                                cnt = cnt + 1;
                                
                            end
                            if mod(cd,4) == 0
                                C1 = C; A1 = A;
                                D1 = C1 + (cd/4)*[-1 1];
                                B1 = A1 + (cd/4)*[-1 1];
                                
                                C2 = D1; A2 = B1;
                                D2 = C2 + (cd/2)*[-1 1];
                                B2 = A2 + (cd/2)*[-1 1];
                                
                                C3 = D2; D3 = D;
                                A3 = B2; B3 = B;
                                
                                rect1 = ABCDtoRect(A1,B1,C1,D1, VEC_SHIFT, IISIZE);
                                rect2 = ABCDtoRect(A2,B2,C2,D2, VEC_SHIFT, IISIZE);
                                rect3 = ABCDtoRect(A3,B3,C3,D3, VEC_SHIFT, IISIZE);
                                rects{cnt} = {rect1, rect2, rect3};
                                cols{cnt} = [1 -1 1];
                                types{cnt} = 'Lienhart45';

                                %rect_vis_ind45(zeros(IISIZE), rects{cnt}, cols{cnt},1);
                                cnt = cnt + 1;
                                
                            end
                            
                            
                            
                            %disp([num2str(camax)]);
                            
                            %rect_vis_ind45(zeros(IISIZE), {rect}, 1,1);
                        end
                    end
                end
            end
           % keyboard;
        end
    end
end

rects = rects(1:cnt-1);
cols = cols(1:cnt-1);
%areas = areas{1:cnt};
types = types(1:cnt-1);


disp(['...defined ' num2str(cnt) ' total Lienhart45 features.']);




function rect = ABCDtoRect(A,B,C,D, VEC_SHIFT, IISIZE)

ROW = [A(1) B(1) C(1) D(1)];
COL = [A(2) B(2) C(2) D(2)];

rect = sub2ind(IISIZE, ROW, COL);
rect = rect + VEC_SHIFT;



function stop = exploreline(IISIZE,start,vec)

r = start(1); c = start(2);
%pr = sign(stop(1) - start(1));  % + indicated increasing from start to stop
%pc = sign(stop(2) - start(1));  % + indicates increasing from start to stop

%while (pr*r <= pr*stop(1)) && (r > 0) && (r <= size(BW,1)) && (pc*c <= pc*stop(2)) && (c > 0) && (c <= size(BW,2))
while (r > 0) && (r <= IISIZE(1)) && (c > 0) && (c <=IISIZE(2))
    stop = [r c];
    r = r+vec(1);
    c = c+vec(2);
end
