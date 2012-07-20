function mv = trkRenderImagesNucleus(TMAX, G, date_txt, num_txt, label_txt, cols, Dlist, tracks, D, DISPLAY_FIGURES, SHOW_FALSE_DETECTS) %#ok<*INUSL>
% 1. draw results on the videos.
% 2. draw text annotations on the image

CONTRAST = 1;  %.4;  %[0,1] 1 is normal contrast, 0 is VERY stretched contrast
RedIntensityThresh  = 1; %200; %280;

for t = 1:TMAX

    I = 1- mat2gray(G{t});
    Ir = I; Ig = I; Ib = I;
    
    %% 1. draw the objects

    % draw nucleus and soma
    for d = 1:length(Dlist{t})
        detect_ind = Dlist{t}(d);

        if tracks(detect_ind) ~= 0


            if D(detect_ind).MeanRedIntensity < RedIntensityThresh
                color = [.6 .6 .6];
            else
                color = cols(tracks(detect_ind),:);
            end
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
    blk = [80 80 80];
    
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
    %I=trkRenderText(I,date_txt, blk, [10, 20], 'bnd2', 'left');
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
