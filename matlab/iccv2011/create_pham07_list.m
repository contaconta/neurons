IMSIZE = [24 24];
%IMSIZE = [34 34];

DISP = 0;

filename = [pwd '/' 'pham' num2str(IMSIZE(1)) 'x' num2str(IMSIZE(2)) '.list'];


R = []; C = []; W = []; T = [];

% generate long diagonal line features
disp('...generating long diagonal features');
[ cols,  r,c, area, weight] = generate_diagonal_features(IMSIZE); %#ok<*ASGLU>
R = [R; r];
C = [C; c];
W = [W; weight];
T = [T; zeros(size(r))];

% generate corner2 features
disp('...generating corner2 features');
[ cols, r,c, area, weight] = generate_corner2_features(IMSIZE);
R = [R; r];
C = [C; c];
W = [W; weight];
T = [T; zeros(size(r))];

% generate corner3 features
disp('...generating corner3 features');
[ cols, r,c, area, weight] = generate_corner3_features(IMSIZE);
R = [R; r];
C = [C; c];
W = [W; weight];
T = [T; zeros(size(r))];

% generate cross features
disp('...generating cross features');
[ cols, r,c, area, weight] = generate_cross_features(IMSIZE);
R = [R; r];
C = [C; c];
W = [W; weight];
T = [T; zeros(size(r))];


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

% generate the surrounding box feature
disp('...generating center surround features');
[rects, cols, types, r,c, area, weight] = generate_center_surround(IMSIZE);
R = [R; r];
C = [C; c];
W = [W; weight];
T = [T; zeros(size(r))];



fid = fopen(filename, 'w');
fprintf(fid, '%d\n', length(R));

disp(['...writing ' num2str(length(R)) ' total features to ' filename]);

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
       end
        
        str = [str strk]; %#ok<AGROW>
        
    end
    
    fprintf(fid, [str '\n']);
    %disp(str);
    %keyboard;
end


fclose(fid);