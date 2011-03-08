clear all; close all; clc;


filename = 'MatchingPursuitsVJ24x24_K';

addpath([pwd '/GaussReconstruction']);


IMSIZE = [24 24];
KMAX = 18;
KMIN = 4;
PAD = 8;
ROTATABLE = 1;
TRANSLATABLE = 1;
DISPLAY = 1;
GLOBAL_OPTIMIZATION = 1;
thre_per = 8e-5;

tmpfile = ['tempfile' num2str(KMAX) '.list'];
disp(['...writing to ' tmpfile]);

if ~GLOBAL_OPTIMIZATION
    filename = [filename int2str(KMAX) 'Greedy.list'];
else
    filename = [filename int2str(KMAX) 'Global.list'];
end

R=[]; C=R; W=R; T=R;

% generate the standard VJ features
disp('...generating standard viola-jones prototypes');
[r,c,rect,col] = generate_viola_jones_prototypes(IMSIZE);
R = [R; r];
C = [C; c];
W = [W; compute_weight(r,c,col)];
T = [T; zeros(size(r))];

BLANK = zeros(IMSIZE);

% feth's init script
generateGaussiansMixturesVJ;

fid = fopen(tmpfile, 'w');
%fprintf(fid, '%d\n', length(R));

count = 1;

for i = 1:length(R)
    Ri = R{i};
    Ci = C{i};
    Wi = W{i};
    Ti = T(i);
    
    RANK = length(Ri);
    str = num2str(RANK);
    
    % create the rect feature vector
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
        str = [str strk]; 
    end
    f = str2num(str); %#ok<ST2NM>

    % create the mask
    M = rectRender(f, IMSIZE, BLANK); 
    
    [r,c] = find(M ~= 0);
    xc = mean(c)-1;
    yc = mean(r)-1;
    
    
    [rinds cinds] = find(M ~= 0);
    r0 = 1;
    c0 = 1;
    r1 = max(rinds);
    c1 = max(cinds);
    Width = c1-c0+1;
    Height = r1-r0+1;
    AREA = Width * Height;
 
    
    
    %     M = M(r0:r1,c0:c1);
    
    MPAD = padarray(M, [PAD PAD]);
    
   
   
    % FIND A VALUE FOR K
    tic;
    Tolerance = thre_per*AREA;
    
    Mixture = MatchingPursuitGaussianApproximationFindNbGaussians(MPAD, Sigmas, Kernels, G, L2_Norms, Tolerance, KMAX);
    K = length(Mixture.Weights);

    if K > AREA
        K = AREA;
    end
    
   	if mod(K,2) == 1
        K = K + 1;
    end
     disp(['K = ' num2str(K)]);
    
    % DO THE RECONSTRUCTION
    if GLOBAL_OPTIMIZATION
        Mixture = ModifiedMatchingPursuitGaussianApproximation(MPAD, Sigmas, Kernels, G, L2_Norms, K);
    else
        Mixture = MatchingPursuitGaussianApproximation_WithCleanUp(MPAD, Sigmas, Kernels, G, L2_Norms, K);
    end
  
    [x y w s] = getMyGaussiansFromFeth(Mixture, PAD, IMSIZE);
    toc;
    
    Nb_Repetitions = prod(IMSIZE - [Width-1, Height-1]);
    
    for iii = 1:Nb_Repetitions
        appendSparseFeature(fid,x,y,w,s,xc,yc,ROTATABLE,TRANSLATABLE,Width,Height);
        count = count + 1;
    end
    
    tic
    if DISPLAY
        if ~mod(i, 1)
            figure(1); imshow(M, [-max(abs(M(:))) max(abs(M(:)))], 'InitialMagnification', 1400);
            xlim([-5 30]);
            ylim([-5 30]);
            R1 = zeros(size(MPAD));

            for j = 1:length(Mixture.Weights)
                R1 = R1 + Mixture.Weights(j)*G{Mixture.Mu(1, j), Mixture.Mu(2, j), Sigmas == Mixture.Sigmas(j)};
            end

            R1 = R1(PAD+1:end-PAD, PAD+1:end-PAD);

            %R1 = reconstruction_coarse(IMSIZE,x,y,w,s,0);
            figure(2);
            imshow(R1, [-max(abs(R1(:))) max(abs(R1(:)))], 'InitialMagnification', 1400);  colormap gray; title(int2str(K));
            xlim([-5 30]);
            ylim([-5 30]);
            hold on;
            plot(x(w > 0)+1, y(w > 0)+1, 'rs');
            plot(x(w < 0)+1, y(w < 0)+1, 'g.'); 
            plot(xc+1,yc+1, 'mo'); hold off;
            drawnow;
        end
    end
    toc
    disp(['feature ' num2str(i) ' / ' num2str(length(R))]);
    
 	pause(0.1);
end

count = count -1;
disp(['...' num2str(count) ' features generated']);
fclose(fid);

disp(['...copying to ' filename]);
fid1 = fopen(tmpfile, 'r');
fid2 = fopen(filename, 'w');
fprintf(fid2, '%d\n', count);
while 1
    tline = fgetl(fid1);
    if ~ischar(tline), break, end
    fprintf(fid2, [tline '\n']);
end
fclose(fid1);
fclose(fid2);
disp(['...wrote to ' filename]);

cmd = ['rm ' tmpfile];
system(cmd);

