IMSIZE = [24 24];
%IMSIZE = [34 34];

DISP = 0;

filename = [pwd '/' 'lienhart' num2str(IMSIZE(1)) 'x' num2str(IMSIZE(2)) '.list'];


R = []; C = []; W = []; T = [];

% generate the standard VJ features
disp('...generating standard viola-jones features');
[r,c,rect,col] = generate_viola_jones_features(IMSIZE); %#ok<ASGLU>
R = [R; r];
C = [C; c];
W = [W; compute_weight(r,c,col)];
T = [T; zeros(size(r))];

% generate the special VJ feature
disp('...generating special viola-jones features');
[r,c,rect,col] = generate_viola_jones_features_special(IMSIZE, 'shapes', 'horz3', 'vert3'); %#ok<ASGLU>
R = [R; r];
C = [C; c];
W = [W; compute_weight(r,c,col)];
T = [T; zeros(size(r))];

% generate the surrounding box features
disp('...generating center surround features');
[rects, cols, types, r,c, area, weight] = generate_center_surround(IMSIZE);
R = [R; r];
C = [C; c];
W = [W; weight];
T = [T; zeros(size(r))];

% generate the 45 degree lienhart features
[rect, col, type, X, Y, weights] = generate_45_features(IMSIZE);
R = [R; Y];
C = [C; X];
W = [W; weights];
T = [T; ones(size(X))];

fid = fopen(filename, 'w');
fprintf(fid, '%d\n', length(R));

disp(['...writing to ' filename]);

for i = 1:length(R)
    Ri = R{i};
    Ci = C{i};
    Wi = W{i};
    Ti = T(i);
    
    RANK = length(Ri);
    str = num2str(RANK);
    
    for k = 1:length(Ri)
       	Rk = Ri{k};
        Ck = Ci{k};
        
        if Ti == 0
          	ru = Rk(1) - 1;
            cu = Ck(1) - 1;
            rl = Rk(2) - 1 -1;
            cl = Ck(2) - 1 -1;
            strk = [' ' num2str(Ti) ' ' num2str(Wi(k)) ' ' num2str(cu) ' ' num2str(ru) ' ' num2str(cl) ' ' num2str(rl)];
        else
            x1 = Ck(1) - 1;
            x2 = Ck(2) - 1;
            x3 = Ck(3) - 1;
            x4 = Ck(4) - 1;
            y1 = Rk(1) - 1;
            y2 = Rk(2) - 1;
            y3 = Rk(3) - 1;
            y4 = Rk(4) - 1;
            strk = [' ' num2str(Ti) ' ' num2str(Wi(k)) ' ' num2str(x1) ' ' num2str(y1) ' ' num2str(x2) ' ' num2str(y2)  ' ' num2str(x3) ' ' num2str(y3)  ' ' num2str(x4) ' ' num2str(y4) ];
        end
        
        str = [str strk]; %#ok<AGROW>
        
    end
    
    fprintf(fid, [str '\n']);
    %disp(str);
    %keyboard;
end


fclose(fid);