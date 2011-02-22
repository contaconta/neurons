clear all;


filename = 'MatchingPursuitsVJGreedy24x24.list';
disp(['...writing to ' filename]);
addpath([pwd '/GaussReconstruction']);


IMSIZE = [24 24];
KMAX = 24;
KMIN = 4;
PAD = 8;
ROTATABLE = 1;
TRANSLATABLE = 1;
DISPLAY = 0;
GLOBAL_OPTIMIZATION = 0;

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

fid = fopen(filename, 'w');
fprintf(fid, '%d\n', length(R));


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
    
    K = KMIN;
   
    % FIND A VALUE FOR K
    tic;
    E = Inf;
    while (E > .12) && (K < KMAX) 
        [Mixture E]= MatchingPursuitGaussianApproximation_WithCleanUp(MPAD, Sigmas, Kernels, G, L2_Norms, K);
        %[E E/AREA K]
        E = E / AREA;
        K = K + 1;
    end
    
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


    appendSparseFeature(fid,x,y,w,s,xc,yc,ROTATABLE,TRANSLATABLE,Width,Height);
    

    if DISPLAY
        figure(1); imagesc(M,[-max(abs(M(:))) max(abs(M(:)))]); colormap gray; 

        R1 = reconstruction_coarse(IMSIZE,x,y,w,s,0);
        figure(2);
        imagesc(R1, [-max(abs(R1(:))) max(abs(R1(:)))]);  colormap gray; title('5sigma + 1');
        hold on;
        plot(x(w > 0)+1, y(w > 0)+1, 'rs');
        plot(x(w < 0)+1, y(w < 0)+1, 'g.'); 
        plot(xc+1,yc+1, 'mo'); hold off;
        drawnow;
    end
    disp(['feature ' num2str(i) ' / ' num2str(length(R))]);
    
 	%pause;
end

disp(['...wrote to ' filename]);
fclose(fid);