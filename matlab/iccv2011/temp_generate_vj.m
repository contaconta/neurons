
WINDOW_SIZE = [24 24];
%WINDOW_SIZE = [34 34];
BlankI = zeros(WINDOW_SIZE);

DISP = 0;

[R,C,I,P] = generate_viola_jones_features(WINDOW_SIZE, 'shapes', {'vert2', 'vert3', 'horz2', 'horz3', 'checker'});

filename = [pwd '/' 'lienhart' num2str(IMSIZE(1)) 'x' num2str(IMSIZE(2)) '.list'];

fid = fopen(filename, 'w');
fprintf(fid, '%d\n', length(R));

for i = 1:length(R)
    
    Ri = R{i};
    Ci = C{i};
    Pi = P{i};
    
    if DISP
        B = BlankI;
    end
    
    RANK = length(Ri);
   	A = zeros(1,RANK);
    W = A;
    
    str = num2str(RANK);
    
    for k = 1:length(Ri)
        Rk = Ri{k};
        Ck = Ci{k};
        
        ru = Rk(1) - 1;
        cu = Ck(1) - 1;
        rl = Rk(2) - 1 -1;
        cl = Ck(2) - 1 -1;
        
        if DISP
            B(ru+1:rl+1,cu+1:cl+1) = Pi(k);
        end
        
        A(k) = (rl+1-ru)*(cl+1-cu);
    end
    
    winds = find(Pi == 1);
    binds = find(Pi == -1);
    
    W(winds) = A(winds) / sum(A(winds));
    W(binds) = -1 * ( A(binds) / sum(A(binds)));
%    disp(num2str(W));
    
    for k = 1:length(Ri)
       	Rk = Ri{k};
        Ck = Ci{k};
        
        ru = Rk(1) - 1;
        cu = Ck(1) - 1;
        rl = Rk(2) - 1 -1;
        cl = Ck(2) - 1 -1;

        strk = [' 0 ' num2str(W(k)) ' ' num2str(cu) ' ' num2str(ru) ' ' num2str(cl) ' ' num2str(rl)];
        
        str = [str strk]; %#ok<AGROW>
        
    end
    
    fprintf(fid, [str '\n']);
    
    if DISP
        disp(str);    
        imagesc(B, [-1 1]); colormap gray;
        drawnow;
        pause(0.005);
    end
%     if mod(i, 300) == 0
%         keyboard;
%     end
end

fclose(fid);
