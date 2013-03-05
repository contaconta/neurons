function mv = trkRenderImagesAndTracks(Green, Cells, CellsList, tracks, num_txt, label_txt, cols ) 


mv = cell(size(Green));

B = zeros(size(Green{1},1), size(Green{1},2));
TMAX = length(Green);
parfor t = 1:TMAX
    %I = mv{t};
    I = double(Green{t});
    I = 1- mat2gray(I);
    Ir = I; Ig = I; Ib = I;
    
    %% 1. draw the objects

    % draw nucleus and soma
    for d = 1:length(CellsList{t})
        detect_ind = CellsList{t}(d);

        if tracks(detect_ind) ~= 0    %#ok
            
            color = cols(tracks(detect_ind),:);    %#ok

            
            % color basic filament skeletons
            currentCell = Cells(detect_ind); %#ok
            FILMASK = currentCell.Neurites; 

            Ir(FILMASK) = color(1);
            Ig(FILMASK) = color(2);
            Ib(FILMASK) = color(3);
            
            
%             numNeurites = currentCell.NeuritesNumber;
%             for i = 1:numNeurites
%                 neuritepixels = FILAMENTS(detect_ind).PixelIdxList( FILAMENTS(detect_ind).NeuriteID == i);
%                 %coloffset = 0.8 * rand(1)  - .4;
%                 coloffset = -.2;
%                 Ir(neuritepixels) = min(1,max(0, color(1) - coloffset));
%                 Ig(neuritepixels) = min(1,max(0, color(2) - coloffset));
%                 Ib(neuritepixels) = min(1,max(0, color(3) - coloffset));
%             end
%             
%             branchpts = FILAMENTS(detect_ind).PixelIdxList( FILAMENTS(detect_ind).NumKids >= 2);
%             leafpts   = FILAMENTS(detect_ind).PixelIdxList( FILAMENTS(detect_ind).NumKids == 0);
%             
%             Ir(branchpts) = 1;
%             Ig(branchpts) = 0;
%             Ib(branchpts) = 0;
%             
%             Ir(leafpts) = 0;
%             Ig(leafpts) = 1;
%             Ib(leafpts) = 0;
            
            % color the soma
            SomaM = B > Inf;
            SomaM(currentCell.SomaPixelIdxList) = 1;
            SomaP = bwmorph(SomaM, 'remove');
            SomaP = bwmorph(SomaP, 'dilate');
            SomaP = bwmorph(SomaP, 'thin',1);
            Ir(SomaP) = max(0, color(1) - .2);
            Ig(SomaP) = max(0, color(2) - .2);
            Ib(SomaP) = max(0, color(3) - .2);

            % color the nucleus
            Ir(currentCell.NucleusPixelIdxList) = color(1);
            Ig(currentCell.NucleusPixelIdxList) = color(2);
            Ib(currentCell.NucleusPixelIdxList) = color(3);

        end
    end


    I(:,:,1) = Ir; I(:,:,2) = Ig; I(:,:,3) = Ib;

    %% 2. render text annotations
    I = uint8(255*I);
    coloroffset = [-.1 -.1 -.1];    % [.2 .2 .2];
    blk = [80 80 80];
    
    for d = 1:length(CellsList{t})
        detect_ind = CellsList{t}(d);

        
        
        % add text annotation
        if tracks(detect_ind) ~= 0
            
            color = cols(tracks(detect_ind),:);
            
            
            col = max(color+coloroffset, 0);
            rloc = max(1,Cells(detect_ind).SomaCentroid(2) - 30);
            cloc = max(1,Cells(detect_ind).SomaCentroid(1) + 20);
            I=trkRenderText(I,['id=' num2str(Cells(detect_ind).ID)], floor(255*col), [rloc, cloc], 'bnd2', 'left');
        end
    end

    % print the name of the experiment on top of the video
    I=trkRenderText(I,num_txt, blk, [10, 175], 'bnd2', 'left');
    I=trkRenderText(I,label_txt, blk, [10, 240], 'bnd2', 'left');
    

    % store the image for writing a movie file
    mv{t} = I;

end