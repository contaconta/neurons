labels = double(imread('../annotated44.png'));
load('../frangi44.mat');

%filaments = labels(:,:,2);
%indexFilaments = find(filaments > 100);

filaments = labels(:,:,2) > 200;
filaments = bwmorph(filaments, 'skel', Inf);
indexFilaments = find(filaments);

maxImg = max(labels, [], 3);
indexBg = find(maxImg < 100);

npos = size(indexFilaments,1);


Rp = FRANG(indexFilaments);
Rp = Rp( find(Rp > 0) );
Rp = log(Rp);
Rn = FRANG(indexBg);
%idx2 = find(Rn > 0.000000001);
%Rn = Rn(idx2);
Rn = log(Rn(find(Rn)));

Rn = randsample(Rn, numel(Rp));

Lp = zeros(size(Rp))+1;
Ln = zeros(size(Rn));

R = [Rp;Rn];
L = [Lp; Ln];

%%
mex PlattSigmoid.cpp CXXFLAGS="\$CXXFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp"
%%
[A,B] = PlattSigmoid(R, L);


%% 
close all;
figure;
[np,xp]=hist(Rp,200);
[nn,xn]=hist(Rn,200);
plot(xp,np,'b');
%plotyy(xp,np,xn,nn);
hold on;
plot(xn,nn,'r');

xmin = min(min(xn), min(xp));
xmax = max(max(xn), max(xp));
step = 0.0001;
scale = max(max(np), max(nn));
x = xmin:step:xmax;
y = scale*1./(1 + exp(A*x + B));
plot(x, y,'k');


figure;
logFRANG = log(FRANG);
%logFRANG(logFRANG == -Inf) = 0;
imagesc(1./(1+exp(A*logFRANG + B)));
