

name = 'D';
ext = '.mat';

load([name ext]);


d2 = size(D,2);

for i = 1:size(D,1)
    
    disp(['...computed 45 degree integral image for ' num2str(i) '/' num2str(size(D,1))]);
   I = ii2image(D(i,1:d2), [24 24], 'outer');
   II45 = single(integralImage45(I));
   II45 = II45(:)';
   
   D(i,d2+1:d2+length(II45)) = II45;
end


save([name '_45' ext], 'D', 'L');