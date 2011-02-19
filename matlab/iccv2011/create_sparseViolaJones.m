clear; 
filename = 'violajones24x24.list';
%filename = 'pham24x24.list';

addpath('/home/ksmith/code/neurons/matlab/iccv2011/anigaussm/');

Sigmas = [.5 1:8];
Kmin = 16;
Kmax = 16;
Afactor = 8;


IMSIZE = [24 24]; BLANK = zeros(IMSIZE);
fid=fopen(filename);
tline = fgetl(fid);

nfeatures = str2double(tline);

outname = ['mpursuitViolaJonesK' num2str(Kmin) '-' num2str(Kmax) '_' num2str(IMSIZE(1)) 'x' num2str(IMSIZE(2)) '.list'];

fid2 = fopen(outname, 'w');
fprintf(fid2, '%d\n', nfeatures);

disp(['...writing to ' outname]);

for i = 1:nfeatures
    tline = fgetl(fid);

    f = str2num(tline); %#ok<ST2NM>
    B = rectRender(f, IMSIZE, BLANK);
    [r,c] = find(B ~= 0);
    XC = mean(c)-1;
    YC = mean(r)-1;
    
    
    % select and appropriate number for K
    %K = determineKfromMask(B, Kmin, Kmax, Afactor);
    K = Kmin;
    
    % mean-center the target signal
    T = B;
    T(T == 1) = T(T ==1) / sum(sum(T == 1));
    T(T == -1) = T(T==-1) / sum(sum(T == -1));
    B = T;
    
    
    % get the matching pursuit approximation
    [X Y W S m] = MatchingPursuitGaussianApproximation(B, Sigmas, K);
    X = X-1;
    Y = Y-1;

    

    n = length(X);
    f = [num2str(n) sprintf(' %f %f', XC, YC)];
    for k = 1:n        
        f = [f sprintf(' %f %f %d %d',  W(k), S(k), X(k), Y(k) )]; %#ok<*AGROW>
    end
    
    
    %keyboard;
    
    if mod(i, 200) == 0
        disp(['feature ' num2str(i) ' (K=' num2str(k) '/X=' num2str(numel(X)) ') samples: ' f]);
        R = sparseRender(str2num(f),IMSIZE); %#ok<ST2NM>
        subplot(2,2,1); imagesc(B); colormap gray; title('original mask');
        subplot(2,2,2); 
        cla; imagesc(R,[-max(abs(R(:))) max(abs(R(:)))]);  colormap gray; hold on; title('analytic');
        plot(X(W > 0)+1, Y(W > 0)+1, 'rs');
        plot(X(W < 0)+1, Y(W < 0)+1, 'g.'); 
        plot(XC+1,YC+1, 'mo'); hold off;
        R2 = reconstruction_coarse(IMSIZE,X,Y,W,S);
        subplot(2,2,3); imagesc(R2, [-max(abs(R2(:))) max(abs(R2(:)))]);  colormap gray; title('2sigma + 1');
        Rfine = reconstruction_coarse(IMSIZE,X,Y,W,S,0);
        subplot(2,2,4); imagesc(Rfine, [-max(abs(Rfine(:))) max(abs(Rfine(:)))]);  colormap gray; title('6sigma + 1');
        drawnow;
        
        [sum(sum(abs(B-B))) sum(sum(abs(R-B))) sum(sum(abs(R2-B))) sum(sum(abs(Rfine-B)))]
        
        %keyboard;
%         [sum(sum(T)) sum(sum(R))/sum(sum(abs(B)))]
        %pause;
    end
    
    fprintf(fid2, [f '\n']);

    
    %keyboard;
    
end

fclose(fid);
fclose(fid2);