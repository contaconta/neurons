function mv = trkRenderImages(Green, Nuclei, Somata) %#ok<*INUSL>

TMAX = length(Green);

parfor t = 1:TMAX

    I = double(Green{t});
    I = 1- mat2gray(I);
    Ir = I; Ig = I; Ib = I;
    
%     Ir(FILAMENTS{t}) = 0;
%     Ig(FILAMENTS{t}) = 0;
%     Ib(FILAMENTS{t}) = 1;
%     
    % color the soma
    for j = 1:max(Somata{t}(:))
        SomaM = Somata{t}==j;
        SomaP = bwmorph(SomaM, 'remove');
        SomaP = bwmorph(SomaP, 'dilate');
        SomaP = bwmorph(SomaP, 'thin',1);
        Ir(SomaP) = 0;
        Ig(SomaP) = 1;
        Ib(SomaP) = 0;
    end
    

    Ir(Nuclei{t} > 0) = 1;
	Ig(Nuclei{t} > 0) = 0;
    Ib(Nuclei{t} > 0) = 0;
    
    
        
    I(:,:,1) = Ir; I(:,:,2) = Ig; I(:,:,3) = Ib;

    %% 2. render text annotations
    I = uint8(255*I);
    % store the image for writing a movie file
    mv{t} = I;
end
% for i = 1:length(Soma)
%     t = Soma(i).Time;
%     SomaM = zeros(size(Nuclei{t}));
%     SomaM(Soma(i).PixelIdxList) = 1;
%     SomaP = bwmorph(SomaM, 'remove');
%     SomaP = bwmorph(SomaP, 'dilate');
%     SomaP = bwmorph(SomaP, 'thin',1);
%     I     =  mv{t};
%     I(SomaP) = 0;
%     I(SomaP) = 255;
%     I(SomaP) = 0;
%     mv{t} = I; 
% end