function mv = trkRenderImages(Green, Nuclei, Somata, FILAMENTS) %#ok<*INUSL>

TMAX = length(Green);

parfor t = 1:TMAX

    I = double(Green{t});
    I = 1- mat2gray(I);
    Ir = I; Ig = I; Ib = I;
    
    Ir(FILAMENTS{t}) = 0;
    Ig(FILAMENTS{t}) = 0;
    Ib(FILAMENTS{t}) = 1;
    
    % color the soma
    SomaM = Somata{t} > 0;
    SomaP = bwmorph(SomaM, 'remove');
    SomaP = bwmorph(SomaP, 'dilate');
    SomaP = bwmorph(SomaP, 'thin',1);
    Ir(SomaP) = 0;
    Ig(SomaP) = 1;
    Ib(SomaP) = 0;

    Ir(Nuclei{t} > 0) = 1;
	Ig(Nuclei{t} > 0) = 0;
    Ib(Nuclei{t} > 0) = 0;
    
    
        
    I(:,:,1) = Ir; I(:,:,2) = Ig; I(:,:,3) = Ib;

    %% 2. render text annotations
    I = uint8(255*I);
    % store the image for writing a movie file
    mv{t} = I;
end