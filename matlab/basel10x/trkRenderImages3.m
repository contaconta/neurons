function mv = trkRenderImages3(TMAX, G, D, Soma, Dlist, branches) %#ok<*INUSL>

B = zeros(size(G{1},1), size(G{1},2));

parfor t = 1:TMAX

    I = double(G{t});
    I = ( 1-((I - min(I(:))) / (max(I(:)) - min(I(:)))));
    Ir = I; Ig = I; Ib = I;
    %% 1. draw the objects

    % draw nucleus and soma
    for d = 1:length(Dlist{t})
        detect_ind = Dlist{t}(d);

        % color the soma
        SomaM = B > Inf;
        SomaM(Soma(detect_ind).PixelIdxList) = 1;%#ok
        branches{t}(Soma(detect_ind).PixelIdxList) = 0;%#ok
        SomaP = bwmorph(SomaM, 'remove');
        SomaP = bwmorph(SomaP, 'dilate');
        SomaP = bwmorph(SomaP, 'thin',1);
        Ir(SomaP) = 0;
        Ig(SomaP) = 1;
        Ib(SomaP) = 0;

        % color the nucleus
        Ir(D(detect_ind).PixelIdxList) = 1;%#ok
        Ig(D(detect_ind).PixelIdxList) = 0;
        Ib(D(detect_ind).PixelIdxList) = 0;

    end
    
    Ir(branches{t}) = 0;
    Ig(branches{t}) = 0;
    Ib(branches{t}) = 1;
    
    
    I(:,:,1) = Ir; I(:,:,2) = Ig; I(:,:,3) = Ib;

    %% 2. render text annotations
    I = uint8(255*I);
    % store the image for writing a movie file
    mv{t} = I;
end