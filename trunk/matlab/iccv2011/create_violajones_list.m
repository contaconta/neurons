IMSIZE = [24 24];
%IMSIZE = [34 34];
%IMSIZE = [20 20];

DISP = 0;

filename = [pwd '/' 'violajones' num2str(IMSIZE(1)) 'x' num2str(IMSIZE(2)) '.list'];


R = []; C = []; W = []; T = [];

% generate the standard VJ features
disp('...generating standard viola-jones features');
[r,c,rect,col] = generate_viola_jones_features(IMSIZE); %#ok<ASGLU>
R = [R; r];
C = [C; c];
W = [W; compute_weight(r,c,col)];
T = [T; zeros(size(r))];




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
        
        end
        
        str = [str strk]; %#ok<AGROW>
        
    end
    
    fprintf(fid, [str '\n']);
end


fclose(fid);