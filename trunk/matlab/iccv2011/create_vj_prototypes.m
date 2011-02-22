clear all;

IMSIZE = [24 24];

NbGaussians  = 16;

addpath([pwd '/GaussReconstruction']);

PAD = 8;

R = []; C = []; W = []; T = [];

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


parfor i = 1:length(R)
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
    
    f = str2num(str); %#ok<ST2NM>
    
    M = rectRender(f, IMSIZE, BLANK); 
    
    [r,c] = find(M ~= 0);
    xc = mean(c)-1;
    yc = mean(r)-1;
    
    [rinds cinds] = find(M ~= 0);
    r0 = 1;
    c0 = 1;
    r1 = max(rinds);
    c1 = max(cinds);
    
    MPAD = padarray(M, [PAD PAD]);
    
    %M = M(r0:r1,c0:c1);
    
    %Mixture1 = MatchingPursuitGaussianApproximation_WithCleanUp(MPAD, Sigmas, Kernels, G, L2_Norms, NbGaussians);
    Mixture2 = ModifiedMatchingPursuitGaussianApproximation(MPAD, Sigmas, Kernels, G, L2_Norms, NbGaussians);
    
    
    
    
%     % put the GMM into a format I like
%     x = Mixture.Mu(2,:) - PAD;
%     x = x(:);
%     y = Mixture.Mu(1,:) - PAD;
%     y = y(:);
%     w = Mixture.Weights;
%     s = Mixture.Sigmas;
% 
%     % remove any gaussians placed outside of the image due to padding
%     badinds1 = (x < 1);
%     badinds2 = (y < 1);
%     badinds3 = (x > IMSIZE(2));
%     badinds4 = (y > IMSIZE(1));
%     inds = ~(badinds1 | badinds2 | badinds3 | badinds4);
%     x = x(inds);
%     y = y(inds);
%     w = w(inds);
%     s = s(inds);

    %[x y w s] = getMyGaussiansFromFeth(Mixture1, PAD, IMSIZE);



    
    tic;
   	[x2 y2 w2 s2] = getMyGaussiansFromFeth(Mixture2, PAD, IMSIZE);
    toc;


    

%     figure(1); imagesc(M,[-max(abs(M(:))) max(abs(M(:)))]); colormap gray; 
%     
%     R1 = reconstruction_coarse(IMSIZE,x,y,w,s,0);
%     figure(2);
%     imagesc(R1, [-max(abs(R1(:))) max(abs(R1(:)))]);  colormap gray; title('5sigma + 1');
%     hold on;
%     plot(x(w > 0)+1, y(w > 0)+1, 'rs');
%     plot(x(w < 0)+1, y(w < 0)+1, 'g.'); 
%     plot(xc+1,yc+1, 'mo'); hold off;
%     
%     R2 = reconstruction_coarse(IMSIZE,x2,y2,w2,s2,0);
%     figure(3);
%     imagesc(R2, [-max(abs(R2(:))) max(abs(R2(:)))]);  colormap gray; title('5sigma + 1');
%     hold on;
%     plot(x2(w2 > 0)+1, y2(w2 > 0)+1, 'rs');
%     plot(x2(w2 < 0)+1, y2(w2 < 0)+1, 'g.'); 
%     plot(xc+1,yc+1, 'mo'); hold off;
%     
%     
%     drawnow;

    disp(['feature ' num2str(i) ' / ' num2str(length(R))]);
    
    %pause(.02);
    
end


