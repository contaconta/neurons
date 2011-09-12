close all;
clear all;

load ../frangi44.mat
A = -1.5524;
B = -31.1269;

Ir = 1./(1+exp(A*log(FRANG)+B)).*0.995;

BORDER = 10;
M = zeros(size(Ir));
M = M > Inf;
M(1:end,1:BORDER) = 1; 
M(1:end,end-BORDER+1:end) = 1;
M(1:BORDER,1:end) = 1; 
M(end-BORDER+1:end,1:end) = 1;

Ir(M)=0;

IMask =  double(imread('nuc44_321.png'));
IMask2 =  double(imread('nuc44_322.png'));


Iret1 = ComputeProbabilityMapShortestPath(Ir, IMask);
Iret2 = ComputeProbabilityMapShortestPath(Ir, IMask2);

%Iret (Iret > 10000) = 10000;
%Iret2 (Iret2 > 10000) = 10000;
%
% Now we combine the probabilities
idx = find(Ir >= 0.1);

P_A = zeros(size(Ir));
P_A(idx) = Iret1(idx)./(Iret1(idx)+Iret2(idx));

P_B = zeros(size(Ir));
P_B(idx) = Iret2(idx)./(Iret1(idx)+Iret2(idx));


%
figure;
subplot(2,3,1)
imagesc(Ir);
title('Original Probability Map');
subplot(2,3,2)
imagesc(Iret1);
title('P(x|A)');
subplot(2,3,3)
imagesc(P_A);
title('P(A|x)');
% colorbar;

subplot(2,3,4)
imagesc(IMask + IMask2);
title('Soma Positions');
subplot(2,3,5)
imagesc(Iret2);
title('P(x|B)');
subplot(2,3,6)
imagesc(P_B);
title('P(B|x)');
% colorbar;
