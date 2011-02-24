clear;
facefolder = '/home/ksmith/data/faces/EPFL-CVLAB_faceDB/test/pos/';
tmpfolder = '/home/ksmith/code/neurons/matlab/iccv2011/slic/tmp/';
addpath(pwd);
%addpath('/home/ksmith/code/neurons/matlab/iccv2011/anigaussm/');
addpath([pwd '/GaussReconstruction/']);

destfolder = '/home/ksmith/code/neurons/matlab/iccv2011/';



% matching pursuits parameters
Sigmas = [.5 1:8];
KMAX = 18;
KMIN = 4;
BLURFLAG = 0;
DISPLAY = 0;
MEANCENTER = 0;
GLOBAL_OPTIMIZATION = 0;
PAD = 8;
thre_per = 8e-5;
ROTATABLE = 0;
TRANSLATABLE = 0;


% superpixel search parameters
SList = [30 50];         % number of superpixels in the image [30 40 50 60]
plist = [ 2 4 8];    % p = spatial coherence of superpixels [1:2:10 11:3:20]

iccvpath = pwd;
cd(tmpfolder);

d = dir([facefolder '*.png']); I = imread([facefolder d(1).name]); IMSIZE = size(I);

if BLURFLAG; blrstr = 'Blur'; else blrstr = []; end;

% feature list file stuff
outname = [destfolder 'superMP' num2str(KMAX) blrstr '_' num2str(IMSIZE(1)) 'x' num2str(IMSIZE(2)) '.list'];
fid2 = fopen(outname, 'w'); %fprintf(fid2, '%d\n', nfeatures);


% feth's init script
generateGaussiansMixturesVJ;

count = 1;

for i = 1:length(d)
    
    facefile = [facefolder d(i).name];
    I = imread(facefile);
    
    

    for s = SList;
        for p = plist


            % run RK's superpixel code
            cmd = ['../slicSuperpixels_ubuntu64bits ' facefile ' ' num2str(s) ' ' num2str(p) ' > null.txt'];
            system(cmd); %display(cmd);
           	pause(0.05);  

            % read the superpixel label file
            [pthstr name ext] = fileparts(facefile); %spfile = [tmpfolder name '_slic.png'];  %S = imread(spfile);
            datfile = [tmpfolder name '.dat'];
            L = readRKLabel(datfile, size(I)); % Lrgb = label2rgb(L, 'jet'); %labelList = unique(L(:));

            if ~isequal(size(L), IMSIZE)
                disp('Problem loading the dat file! Try to do it again.');
                keyboard;
            end
            
            maxL = max(max(L));

            prop = regionprops(L, 'Centroid');

            %N = round(maxL / 8);
            if p > 10 
                N = 1;
            elseif p < 3
                N = 1;
            else
                N = 1;
            end
            
            for n = 1:N
                
                % select a first superpixel
                ind(1) = randsample(1:maxL,1);
                
                % compute distances to selected superpixel
                dist = zeros(maxL,1);
                for l=1:maxL
                    dist(l) = sqrt( (prop(l).Centroid(1)-prop(ind(1)).Centroid(1))^2 + (prop(l).Centroid(2)-prop(ind(1)).Centroid(2))^2);
                end
                dist(ind(1)) = Inf;
                
                % select a second superpixel
                ind(2) = randsample(1:maxL, 1, true, 1./(dist.^3));


                % display what we selected
                Ir = I; Ig = I; Ib = I;
                Ig(L == ind(1)) = 255; Ir(L==ind(1))=0; Ib(L==ind(1))=0;
                Ig(L == ind(2)) = 0; Ir(L==ind(2))=0; Ib(L==ind(2))=255;
                Irgb(:,:,1) = Ir; Irgb(:,:,2) = Ig; Irgb(:,:,3) = Ib;

                
                % create a binary mask
                M = zeros(size(L));
                M(L == ind(1)) = 1;
                M(L == ind(2)) = -1;
                
                % mean-center the desired signal
                if MEANCENTER
                    M(M == 1) = M(M == 1) ./ sum(sum(M == 1));
                    M(M == -1) = M(M == -1) ./ sum(sum(M == -1));
                end
                    
                [r,c] = find(M ~= 0);
                XC = mean(c)-1;
                YC = mean(r)-1;
                
                % blur the desired mask to make it easier to approximate
                if BLURFLAG
                    M = imgaussian(M,.5);
                    M = M - mean(M(:));
                end
                
                AREA = sum(sum(M ~= 0));
 
    

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

                [X Y W S] = getMyGaussiansFromFeth(Mixture, PAD, IMSIZE);
                toc;

                Width = -1; Height = -1;
               	f = appendSparseFeature(fid2,X,Y,W,S,XC,YC,ROTATABLE,TRANSLATABLE,Width,Height);
             
                
                
                %[X Y W S] = distanceTransformApproximationV2(M, Sigmas, K);
                %[X Y W S] =distanceTransformApproximation(M,Sigmas, K);
                %[X Y W S m] = MatchingPursuitGaussianApproximation(M, Sigmas, K);
%                 X = X-1;
%                 Y = Y-1;

                % append the feature to the output file
%              	f = appendSparseFeature(fid2,X,Y,W,S,XC,YC,0,0);
                
                
                %% display
                if mod(count, 1) == 0
                    %disp(outname);
                    disp([' face (' num2str(i) '/' num2str(length(d)) ') count = ' num2str(count) '  K = ' num2str(K) ' p = ' num2str(p) ' s = ' num2str(s) '  #superpixels = ' num2str(maxL)  ]);
                    
                    if DISPLAY
                        figure(1); clf; cla; imshow(Irgb); set(gca, 'Position', [0 0 1 1]);
                        
                        hold on; h=drawLabels(L); hold off;
                        R = sparseRender(str2num(f),size(L)); %#ok<ST2NM>
                        figure(2);
                        subplot(1,2,1); imagesc(M,[-max(abs(M(:))) max(abs(M(:)))]); colormap gray;
                        subplot(1,2,2); 
                        cla; imagesc(R,[-max(abs(R(:))) max(abs(R(:)))]);  colormap gray; hold on;
                        plot(X(W > 0)+1, Y(W > 0)+1, 'rs');
                        plot(X(W < 0)+1, Y(W < 0)+1, 'g.'); 
                        plot(XC+1,YC+1, 'mo'); hold off;
                    end
                    
                    
                    
                    %pause;
                end
                
                %% clean up
                count = count + 1;


             	%keyboard;
            end
        end
    end
end

disp('=======================================');
disp(' ');
disp('IMPORTANT!!!!');
disp(' ');
disp(['WRITE THE NUMBER OF FEATURES (' num2str(count -1) ') AT THE TOP OF']);
disp(outname);
disp(' ');
disp('=======================================');

fclose(fid2);

cd(iccvpath);



%         if exist('h', 'var')
%             for k = 1:length(h)
%                 delete(h(k));
%             end
%             clear h;
%         end