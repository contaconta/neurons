function mv = trkRenderImages(TMAX, G, date_txt, num_txt, label_txt, SMASK, cols, mv, Dlist, BLANK, FILAMENTS, Soma, tracks, D, DISPLAY_FIGURES, SHOW_FALSE_DETECTS) %#ok<*INUSL>
% 1. draw results on the videos.
% 2. draw text annotations on the image

CONTRAST = 0.4;  %[0,1] 1 is normal contrast, 0 is VERY stretched contrast
RedIntensityThresh  = 250; %280;


B = zeros(size(G{1},1), size(G{1},2));

for t = 1:TMAX

    %I = mv{t};
    I = imadjust(mv{t}, [0; CONTRAST], [1; 0]);
    Ir = I(:,:,1); Ig = I(:,:,2); Ib = I(:,:,3);

    
    %% 1. draw the objects

    % draw nucleus and soma
    for d = 1:length(Dlist{t})
        detect_ind = Dlist{t}(d);

        if tracks(detect_ind) ~= 0

            % fully color the filaments
            %FILMASK = FIL{t} == detect_ind;
            %Ir(FILMASK) = max(0, color(1) - .3);
            %Ig(FILMASK) = max(0, color(2) - .3);
            %Ib(FILMASK) = max(0, color(3) - .3);

            if D(detect_ind).MeanRedIntensity < RedIntensityThresh
                color = [.6 .6 .6];
            else
                color = cols(tracks(detect_ind),:);
            end
            
            % color filament skeletons
            FILMASK = BLANK > Inf;
            FILMASK( FILAMENTS(detect_ind).PixelIdxList) = 1;
            %FILMASK(SMASK{t} > 0) = 0;
            FILMASK(Soma(detect_ind).PixelIdxList) = 0;
            Ir(FILMASK) = max(0, color(1) - .2);
            Ig(FILMASK) = max(0, color(2) - .2);
            Ib(FILMASK) = max(0, color(3) - .2);
            
            % show the filament endpoints
            ENDMASK = BLANK > Inf;
            ENDMASK( FILAMENTS(detect_ind).Endpoints) = 1;
            ENDMASK( Soma(detect_ind).PixelIdxList) = 0;
            Ir( ENDMASK ) = .9;
            Ig( ENDMASK ) = .1;
            Ib( ENDMASK ) = .1;
            
            % show the filament branch points
            BRANCHMASK = BLANK > Inf;
            BRANCHMASK( FILAMENTS(detect_ind).Branchpoints) = 1;
            BRANCHMASK( Soma(detect_ind).PixelIdxList) = 0;
            Ir( BRANCHMASK ) = 1;
            Ig( BRANCHMASK) = 1;
            Ib( BRANCHMASK ) = .1;

            % color the soma
            SomaM = B > Inf;
            SomaM(Soma(detect_ind).PixelIdxList) = 1;
            SomaP = bwmorph(SomaM, 'remove');
            SomaP = bwmorph(SomaP, 'dilate');
            SomaP = bwmorph(SomaP, 'thin',1);
            Ir(SomaP) = max(0, color(1) - .2);
            Ig(SomaP) = max(0, color(2) - .2);
            Ib(SomaP) = max(0, color(3) - .2);

            % color the nucleus
            Ir(D(detect_ind).PixelIdxList) = color(1);
            Ig(D(detect_ind).PixelIdxList) = color(2);
            Ib(D(detect_ind).PixelIdxList) = color(3);



        else
            if SHOW_FALSE_DETECTS
                % color the false detections!
                Ir(D(detect_ind).PixelIdxList) = 1;
                Ig(D(detect_ind).PixelIdxList) = 0;
                Ib(D(detect_ind).PixelIdxList) = 0;
            end
        end
    end


    I(:,:,1) = Ir; I(:,:,2) = Ig; I(:,:,3) = Ib;

    %% 2. render text annotations
    I = uint8(255*I);
    coloroffset = [-.1 -.1 -.1];    % [.2 .2 .2];
    blk = [0 0 0];
    
    for d = 1:length(Dlist{t})
        detect_ind = Dlist{t}(d);

        
        
        % add text annotation
        if tracks(detect_ind) ~= 0
            if D(detect_ind).MeanRedIntensity < RedIntensityThresh
                color = [.6 .6 .6];
            else
                color = cols(tracks(detect_ind),:);
            end
            
            col = max(color+coloroffset, 0);
            rloc = max(1,D(detect_ind).Centroid(2) - 30);
            cloc = max(1,D(detect_ind).Centroid(1) + 20);
            I=trkRenderText(I,['id=' num2str(D(detect_ind).ID)], floor(255*col), [rloc, cloc], 'bnd2', 'left');
        else
            if SHOW_FALSE_DETECTS
                rloc = max(1,D(detect_ind).Centroid(2) - 30);
                cloc = max(1,D(detect_ind).Centroid(1) + 20);
                I=trkRenderText(I,'false_detection', [255 0 0], [rloc, cloc], 'bnd', 'left');
            end
        end
    end

    % print the name of the experiment on top of the video
    
    I=trkRenderText(I,date_txt, blk, [10, 20], 'bnd2', 'left');
    I=trkRenderText(I,num_txt, blk, [10, 180], 'bnd2', 'left');
    I=trkRenderText(I,label_txt, blk, [10, 240], 'bnd2', 'left');

    % show the image
    if DISPLAY_FIGURES
        imshow(I);     pause(0.05);
    end

    % store the image for writing a movie file
    mv{t} = I;

end
