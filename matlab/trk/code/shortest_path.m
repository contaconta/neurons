close all;
clear all;

load Ifloat.mat
figure
imshow(Ir);
Ir = Ir.*0.995;



IMask = zeros(size(Ir));
for x = 365:1:374
    for y = 123:1:133
      IMask(x, y) = 1;
    end
end

IMask2 = zeros(size(Ir));

for x = 200:1:208
   for y = 444:1:452
     IMask2(x, y) = 2;
   end
end

close all;
%mex ComputeProbabilityMapShortestPath.cpp

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
