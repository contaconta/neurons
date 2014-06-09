function mv = trkConvertTo8bit(Images, invert)




mv = cell(size(Images));

TMAX = length(Images);
for t = 1:TMAX
    
    % convert the image
    I = double(Images{t});
    
    if invert
        I = 1- mat2gray(I);
    end
%     Ir = I; Ig = I; Ib = I;
%     
%     
%     Iout(:,:,1) = Ir(7:end-7, 7:end-7); 
%     Iout(:,:,2) = Ig(7:end-7, 7:end-7); 
%     Iout(:,:,3) = Ib(7:end-7, 7:end-7);
%   

	I2 = I - min(I(:));
	I2 = I2 * (255/max(I2(:)));
    I2 =  round(I2);
    Iout = uint8(I2);
    
    Iout = Iout(7:end-7, 7:end-7); 
    
    mv{t} = Iout;
end