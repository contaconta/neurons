%function mv = trkRenderImages3(TMIN,TMAX,date_txt, num_txt,label_txt,cols,mv,Dlist,BLANK,FILAMENTS,Soma,tracks,D,N,DISPLAY_FIGURES) 
function mv = trkRenderImages3(TMIN,TMAX,R,cols,mv,DISPLAY_FIGURES)
date_txt    = R.GlobalMeasures.Date;
num_txt     = R.GlobalMeasures.AssayPosition;
label_txt   = R.GlobalMeasures.Label;
Dlist       = R.Dlist;
BLANK       = zeros(size(mv{1}));
FILAMENTS   = R.FILAMENTS;
Soma        = R.Soma;
tracks      = R.tracks;
D           = R.D;
N           = R.N;

CONTRAST = 0.4;  %[0,1] 1 is normal contrast, 0 is VERY stretched contrast

B = zeros(size(mv{1},1), size(mv{1},2));

for t = TMIN:TMAX
    I = imadjust(mv{t}, [0; CONTRAST], [1; 0]);
    Ir = I(:,:,1); Ig = I(:,:,2); Ib = I(:,:,3);

    %% 1. draw the objects

    % draw nucleus and soma
    for d = 1:length(Dlist{t})
        detect_ind = Dlist{t}(d);
        
        if tracks(detect_ind) ~= 0
            if isfield(D, 'Happy')
                if D(detect_ind).Happy == 0
                    color = [.5 .5 .5];
                else
                    color = cols(tracks(detect_ind),:);
                    color = hsv2rgb(rgb2hsv(color) + [0 -.25 -.1]);
                end
            else
                color = cols(tracks(detect_ind),:);
                color = hsv2rgb(rgb2hsv(color) + [0 -.25 -.1]);
            end

            % color basic filament skeletons
            FILMASK = BLANK > Inf;
            FILMASK( FILAMENTS(detect_ind).PixelIdxList) = 1;
            FILMASK(Soma(detect_ind).PixelIdxList) = 0;
            Ir(FILMASK) = max(0, color(1) - .2);
            Ig(FILMASK) = max(0, color(2) - .2);
            Ib(FILMASK) = max(0, color(3) - .2);

            numNeurites = max(FILAMENTS(detect_ind).NeuriteID);
            for i = 1:numNeurites
%                 if i == 1
%                     neuritecolor = [.6 .6 .6];
%                 elseif i == 2
%                     neuritecolor = [.8 .3 .3];
%                 else
%                     neuritecolor = [ .6 .6 .6];
%                 end
                n = FILAMENTS(detect_ind).NIdxList(i);
                if N(n).NeuriteTrack == 0
                    neuritecolor = [.6 .6 .6];
                else
                    nTrack = N(n).NeuriteTrack;
                    neuritecolor = cols(nTrack,:);
                end
    
                neuritepixels = FILAMENTS(detect_ind).PixelIdxList( FILAMENTS(detect_ind).NeuriteID == i);
                Ir(neuritepixels) = neuritecolor(1);
                Ig(neuritepixels) = neuritecolor(2);
                Ib(neuritepixels) = neuritecolor(3);
            end
            
            % draw filopodia
%             filohsv = rgb2hsv(color) + [.085 0  .075];
%             filohsv(filohsv > 1) = filohsv(filohsv > 1) - 1;
%             filocolor = hsv2rgb(filohsv);
%             [Ir Ig Ib] = colorNeurites(detect_ind, Ir,Ig,Ib,FILAMENTS,filocolor);
            
%             % draw branch points
%             branchpts = FILAMENTS(detect_ind).PixelIdxList( FILAMENTS(detect_ind).NumKids >= 2);
%             branchcolor = hsv2rgb(rgb2hsv(color) + [0 0 .3]);
%             Ir(branchpts) = branchcolor(1);
%             Ig(branchpts) = branchcolor(2);
%             Ib(branchpts) = branchcolor(3);
            
            % color the soma
            SomaM = B > Inf;
            SomaM(Soma(detect_ind).PixelIdxList) = 1;
            SomaP = bwmorph(SomaM, 'remove');
            SomaP = bwmorph(SomaP, 'dilate');
            SomaP = bwmorph(SomaP, 'thin',1);
            somacolor = hsv2rgb(rgb2hsv(color) + [0 0 0]);
            Ir(SomaP) = somacolor(1);
            Ig(SomaP) = somacolor(2);
            Ib(SomaP) = somacolor(3);

            % color the nucleus
            Ir(D(detect_ind).PixelIdxList) = somacolor(1);
            Ig(D(detect_ind).PixelIdxList) = somacolor(2);
            Ib(D(detect_ind).PixelIdxList) = somacolor(3);
        end
    end

    I(:,:,1) = Ir; I(:,:,2) = Ig; I(:,:,3) = Ib;

    %% 2. render text annotations
    I = uint8(255*I);
    blk = [80 80 80];
    
    for d = 1:length(Dlist{t})
        detect_ind = Dlist{t}(d);

        % add text annotation
        if tracks(detect_ind) ~= 0
            if isfield(D, 'Happy')
                if D(detect_ind).Happy == 0
                    color = [.7 .7 .7];
                else
                    color = cols(tracks(detect_ind),:);
                    color = hsv2rgb(rgb2hsv(color) + [0 -.25 -.1]);
                end
            else
                color = cols(tracks(detect_ind),:);
                color = hsv2rgb(rgb2hsv(color) + [0 -.25 -.1]);
            end
            
            col = max(color, 0);
            rloc = max(1,D(detect_ind).Centroid(2) - 30);
            cloc = max(1,D(detect_ind).Centroid(1) + 20);
            I=trkRenderText(I,['id=' num2str(D(detect_ind).ID)], floor(255*col), [rloc, cloc], 'bnd2', 'left');
        end
    end

    % print the name of the experiment on top of the video
    I=trkRenderText(I,date_txt, blk, [10, 20], 'bnd2', 'left');
    I=trkRenderText(I,num_txt, blk, [10, 175], 'bnd2', 'left');
    I=trkRenderText(I,label_txt, blk, [10, 240], 'bnd2', 'left');
    
    % show the image
    if DISPLAY_FIGURES
        imshow(I);
        drawnow;
    end

    % store the image for writing a movie file
    mv{t} = I;
end





%             % draw leaf points
%             leafpts   = FILAMENTS(detect_ind).PixelIdxList( FILAMENTS(detect_ind).NumKids == 0);
%             Ir(leafpts) = 0;
%             Ig(leafpts) = 1;
%             Ib(leafpts) = 0;




function [Ir Ig Ib] = colorNeurites(dlist, Ir,Ig,Ib,FILAMENTS,color)

for n = dlist(:)'
    if ~isfield(FILAMENTS, 'FilopodiaFlag')
        [FILAMENTS neuritePixList] = trkFindNeurites(n,FILAMENTS);
    else
        neuriteIdxList = find(FILAMENTS(n).FilopodiaFlag);
        neuritePixList = FILAMENTS(n).PixelIdxList(neuriteIdxList); %#ok<FNDSB>
    end
        
    Ir(neuritePixList) = color(1);
    Ig(neuritePixList) = color(2);
    Ib(neuritePixList) = color(3);
end

