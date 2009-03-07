clear IStack;

% nuclei = ada_trainingfiles('nuclei24.txt', 'train', '+', 500);
%mito = ada_trainingfiles('mitochondria24.txt', 'train', '+', 500);
% nonnuclei = ada_trainingfiles('nuclei24.txt', 'train', '-', 500);
%nonmito = ada_trainingfiles('mitochondria24.txt', 'train', '-', 500);
%mito = ada_trainingfiles('persons24x64.txt', 'train', '+', 500);
%nonmito = ada_trainingfiles('persons24x64.txt', 'train', '-', 500);
faces = ada_trainingfiles('persons48x128.txt', 'train', '+', 500);
nonfaces = ada_trainingfiles('persons48x128.txt', 'train', '-', 500);
%faces = ada_trainingfiles('faces.txt', 'train', '+', 500);
%nonfaces = ada_trainingfiles('faces.txt', 'train', '-', 500);

POS_EXAMPLES = 3;
NEG_EXAMPLES = 2;

% nuc_inds = ceil(length(nuclei) * rand([N_EXAMPLES 1]));
% mito_inds = ceil(length(mito) * rand([N_EXAMPLES 1]));
% nonnuc_inds = ceil(length(nonnuclei) * rand([N_EXAMPLES 1]));
% nonmito_inds = ceil(length(nonmito) * rand([N_EXAMPLES 1]));
face_inds = ceil(length(faces) * rand([POS_EXAMPLES 1]));
nonface_inds = ceil(length(nonfaces) * rand([NEG_EXAMPLES 1]));

% files = {nuclei{nuc_inds}  mito{mito_inds} nonnuclei{nonnuc_inds}  nonmito{nonmito_inds}}';
%files = {mito{mito_inds} nonmito{nonmito_inds}}';
files = {faces{face_inds} nonfaces{nonface_inds}}';

IStack = [];

for f = 1:length(files)

    I = imread(files{f});
    cls = class(I);
    if length(size(I)) > 2;  I = rgb2gray(I); end;
    I = mat2gray(I, [0 double(intmax(cls))]); 
    
    %I = imnormalize('image', I);
    
    % Image inversion
    %I = -I + 1;
    
    % show the image as the first column in the montage
    if f == 1
        IStack = I;
    else
        IStack(:,:,1,size(IStack,4)+1) = I;
    end
    
%     % show the simple gradient
%     [NORM, ANGL] = simple_grad(I);
%     IStack(:,:,1,size(IStack,4)+1) = NORM;
%     
    
%     EDGE = NORM > .05;
%     IStack(:,:,1,size(IStack,4)+1) = EDGE;
%     
%     EDGE = NORM > .1;
%     IStack(:,:,1,size(IStack,4)+1) = EDGE;
%     
%     EDGE = NORM > .13;
%     IStack(:,:,1,size(IStack,4)+1) = EDGE;
%     
%     EDGE = NORM > .16;
%     IStack(:,:,1,size(IStack,4)+1) = EDGE;
%     
%     EDGE = NORM > .2;
%     IStack(:,:,1,size(IStack,4)+1) = EDGE;
%     
%     EDGE = NORM > .22;
%     IStack(:,:,1,size(IStack,4)+1) = EDGE;
    
%     % show log sigma = 1
%     SIGMA = 1;
%     N=ceil(SIGMA*3)*2+1;
%     h = fspecial('log', [N N], SIGMA);
%     A = 3*imfilter(I, h, 'symmetric');
%     IStack(:,:,1,size(IStack,4)+1) = A+.5;
%     
%     EDGE = edge(I, 'log', 0, SIGMA); %EDGE = bwmorph(EDGE, 'diag');
%     IStack(:,:,1,size(IStack,4)+1) = EDGE;
    
%     % show log sigma = 1.5
%     SIGMA = 1.5;
%     N=ceil(SIGMA*3)*2+1;
%     h = fspecial('log', [N N], SIGMA);
%     A = 4*imfilter(I, h, 'symmetric');
%     IStack(:,:,1,size(IStack,4)+1) = A+.5;
%     
%     EDGE = edge(I, 'log', 0, SIGMA); %EDGE = bwmorph(EDGE, 'diag');
%     IStack(:,:,1,size(IStack,4)+1) = EDGE;
    
    % show log sigma = 2
    SIGMA = 2;
    N=ceil(SIGMA*3)*2+1;
    h = fspecial('log', [N N], SIGMA);
    A = 4*imfilter(I, h, 'symmetric');
    IStack(:,:,1,size(IStack,4)+1) = A+.5;
    
    EDGE = edge(I, 'log', 0, SIGMA); %EDGE = bwmorph(EDGE, 'diag');
    IStack(:,:,1,size(IStack,4)+1) = EDGE;
    
%     EDGE = edge(I, 'log', .007, SIGMA); %EDGE = bwmorph(EDGE, 'diag');
%     IStack(:,:,1,size(IStack,4)+1) = EDGE;

    % show log sigma = 2.5
    SIGMA = 2.5;
    N=ceil(SIGMA*3)*2+1;
    h = fspecial('log', [N N], SIGMA);
    A = 5*imfilter(I, h, 'symmetric');
    IStack(:,:,1,size(IStack,4)+1) = A+.5;
    
    EDGE = edge(I, 'log', 0, SIGMA); %EDGE = bwmorph(EDGE, 'diag');
    IStack(:,:,1,size(IStack,4)+1) = EDGE;

    % show log sigma = 3
    SIGMA = 4;
    N=ceil(SIGMA*3)*2+1;
    h = fspecial('log', [N N], SIGMA);
    A = 5*imfilter(I, h, 'symmetric');
    IStack(:,:,1,size(IStack,4)+1) = A+.5;
    
    EDGE = edge(I, 'log', 0, SIGMA); %EDGE = bwmorph(EDGE, 'diag');
    IStack(:,:,1,size(IStack,4)+1) = EDGE;
    
%     % show prewitt
%     filtX = fspecial('prewitt');
%     filtY = filtX';
%     GradX = imfilter(I,filtX, 'symmetric');
%     GradY = imfilter(I,filtY, 'symmetric');
%     NORM = arrayfun(@(a,b)(norm([a b])), GradX, GradY);
%     NORM = mat2gray(NORM, [min(NORM(:)) max(NORM(:))]);
%     IStack(:,:,1,size(IStack,4)+1) = NORM;
%     
%     EDGE = edge(I,'prewitt', .1);
%     IStack(:,:,1,size(IStack,4)+1) = EDGE;
%     
%     EDGE = edge(I,'prewitt', .2);
%     IStack(:,:,1,size(IStack,4)+1) = EDGE;
    
%     EDGE = edge(I,'prewitt', .06);
%     IStack(:,:,1,size(IStack,4)+1) = EDGE;
%     
%     EDGE = edge(I,'prewitt', .1);
%     IStack(:,:,1,size(IStack,4)+1) = EDGE;
    
    % show sobel
    gh = imfilter(I,fspecial('sobel')' /8,'replicate');
    gv = imfilter(I,fspecial('sobel')/8,'replicate');
    NORM = arrayfun(@(a,b)(norm([a b])), gh, gv);
    NORM = mat2gray(NORM, [min(NORM(:)) max(NORM(:))]);
    IStack(:,:,1,size(IStack,4)+1) = NORM;
    
    EDGE = edge(I,'sobel', .06);
    IStack(:,:,1,size(IStack,4)+1) = EDGE;
    
    EDGE = edge(I,'sobel', .075);
    IStack(:,:,1,size(IStack,4)+1) = EDGE;

    EDGE = edge(I,'sobel', .1);
    IStack(:,:,1,size(IStack,4)+1) = EDGE;
    
    EDGE = edge(I,'sobel', .15);
    IStack(:,:,1,size(IStack,4)+1) = EDGE;
    
   

%     % show canny
%     sigma = 0.5;
%     filtX = gauss0(sigma);
%     filtY = filtX';
%     GradX = imfilter(I,filtX, 'symmetric');
%     GradY = imfilter(I,filtY, 'symmetric');
%     NORM = arrayfun(@(a,b)(norm([a b])), GradX, GradY);
%     IStack(:,:,1,size(IStack,4)+1) = NORM;
%     
%     EDGE = edge(I,'canny', .1 ,sigma);
%     IStack(:,:,1,size(IStack,4)+1) = EDGE;
%     
%     EDGE = edge(I,'canny', .3 ,sigma);
%     IStack(:,:,1,size(IStack,4)+1) = EDGE;
%     
%     EDGE = edge(I,'canny', .5 ,sigma);
%     IStack(:,:,1,size(IStack,4)+1) = EDGE;
%     
%     EDGE = edge(I,'canny', .7 ,sigma);
%     IStack(:,:,1,size(IStack,4)+1) = EDGE;
%     
%     EDGE = edge(I,'canny', .9 ,sigma);
%     IStack(:,:,1,size(IStack,4)+1) = EDGE;
%     
    % show canny
    sigma = .8;
    filtX = gauss0(sigma);
    filtY = filtX';
    GradX = imfilter(I,filtX, 'symmetric');
    GradY = imfilter(I,filtY, 'symmetric');
    gall = [GradX(:) GradY(:)];
    NORM = arrayfun(@(a,b)(norm([a b])), GradX, GradY);
    IStack(:,:,1,size(IStack,4)+1) = NORM;
    
    EDGE = edge(I,'canny');
    IStack(:,:,1,size(IStack,4)+1) = EDGE;
    
    EDGE = edge(I,'canny', .3 ,sigma);
    IStack(:,:,1,size(IStack,4)+1) = EDGE;
    
    EDGE = edge(I,'canny', .5 ,sigma);
    IStack(:,:,1,size(IStack,4)+1) = EDGE;
    
    EDGE = edge(I,'canny', .7 ,sigma);
    IStack(:,:,1,size(IStack,4)+1) = EDGE;
    
%     EDGE = edge(I,'canny', .9 ,sigma);
%     IStack(:,:,1,size(IStack,4)+1) = EDGE;
    
    % show canny
    sigma = 1.5;
    filtX = gauss0(sigma);
    filtY = filtX';
    GradX = imfilter(I,filtX, 'symmetric');
    GradY = imfilter(I,filtY, 'symmetric');
    gall = [GradX(:) GradY(:)];
    NORM = arrayfun(@(a,b)(norm([a b])), GradX, GradY);
    IStack(:,:,1,size(IStack,4)+1) = NORM;
    
    EDGE = edge(I,'canny', .3 ,sigma);
    IStack(:,:,1,size(IStack,4)+1) = EDGE;
    
    EDGE = edge(I,'canny', .5 ,sigma);
    IStack(:,:,1,size(IStack,4)+1) = EDGE;
    
    EDGE = edge(I,'canny', .7 ,sigma);
    IStack(:,:,1,size(IStack,4)+1) = EDGE;
    
%     EDGE = edge(I,'canny', .9 ,sigma);
%     IStack(:,:,1,size(IStack,4)+1) = EDGE;
    
end


montage(IStack, 'Size', [length(files) 21]);

%montage(IStack, 'Size', [2 5]);