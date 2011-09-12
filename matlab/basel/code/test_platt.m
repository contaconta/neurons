nSamples = 100000;

Rp = 5*(randn(nSamples,1)+0.5);
Rp = Rp(:);
Rn = randn(5*nSamples,1)-1.5;
Rn = Rn(:);
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
hold on;
plot(xn,nn,'r');

xmin = min(min(xn), min(xp));
xmax = max(max(xn), max(xp));
step = 0.0001;
scale = max(max(np), max(nn));
x = xmin:step:xmax;
y = scale*1./(1 + exp(A*x + B));
plot(x, y,'k');

