function mv = trkRenderFancy(Images, Cells, CellsList, tracks, cols, mode) 

% define the rendering mode!
if ~exist('mode', 'var')
    mode = 0;
end
% mode 0    final rendering, with text annotations 
% mode 1    final rendering, no text annotations
% mode 2    no neurites, no text
% mode 3    no neurites, no text, blue fill
% mode 4    cell body fill
% mode 5    cell body with candidate endpoints + neurites
% mode 6    binary b&w segmentation
% mode 7    nuclei detections only, red fill
% mode 8    binary b&w segmentation but with cell colors
% mode 9    binary b&w segmentation, no neurites
% mode 10   cell body with candidate endpoints
% mode 11   neurite backtracing
% mode 12   soma growth, red fill

fprintf('rendering mode %d\n', mode);

SomaFaceAlpha = .25;
SomaEdgeAlpha =  1;
NucleusFaceAlpha = 1;
NucleusEdgeAlpha = 0;
WholeFaceAlpha = .25;
WholeEdgeAlpha = 1;
EDGEWIDTH = 1;



f = figure;

mv = cell(size(Images));

B = zeros(size(Images{1},1), size(Images{1},2));
TMAX = length(Images);
for t = 1:TMAX
    
    % convert the image
    if numel(size(Images{t})) == 2
        I = double(Images{t});
        I = 1- mat2gray(I);
        Ir = I; Ig = I; Ib = I;
    else
        I = Images{t};
        Ir = I(:,:,1); Ig = I(:,:,2); Ib = I(:,:,3);
    end
    
    switch mode
        case {6,8,9}
            I = ones(size(I));
            Ir = I; Ig = I; Ib = I;
    end
    
    %% 0. render the scene
    hold off; cla;
    cla; gtp = []; estp = [];
    set(gca, 'Position', [0 0 1 1]);
    imshow(I); hold on;
    
    
    
    %% 1. draw objects on top of the scene
        Soma = [];
        Nuc = [];
        Whole = [];
    
    % compute size of nucleus and soma
    for d = 1:length(CellsList{t})
        detect_ind = CellsList{t}(d);

        if tracks(detect_ind) ~= 0    
            currentCell = Cells(detect_ind);
            
%             color = cols(tracks(detect_ind),:);    
%             switch mode
%                 case 3
%                     NeuriteColor = color;
%                 otherwise
%                     NeuriteColor = color;
%             end
%             
%             % color neurites or the cell body
%             switch mode
%                 case {0,1}
%                     FILMASK = currentCell.Neurites; 
%                     Ir(FILMASK) = NeuriteColor(1);
%                     Ig(FILMASK) = NeuriteColor(2);
%                     Ib(FILMASK) = NeuriteColor(3);
%                 otherwise
%             end
            
            
            
            
            % get soma and nucleus polys
            B = zeros(size(Ir));
            pixlist = currentCell.SomaPixelIdxList;
            B(pixlist) = 1;
            BW = bwboundaries(B, 8, 'noholes');            
            x1 = BW{1}(:,2);
            y1 = BW{1}(:,1);
            [x2, y2] = poly2cw(x1, y1);
            Soma(d).x = x2;
            Soma(d).y = y2;
            
            B = zeros(size(Ir));
            pixlist = currentCell.NucleusPixelIdxList;
            B(pixlist) = 1;
            B = bwmorph(B, 'thin');       % shrink the nucleus by 1 px
            BW = bwboundaries(B, 8, 'noholes');   
            x1 = BW{1}(:,2);
            y1 = BW{1}(:,1);
            [x2, y2] = poly2cw(x1, y1);
            Nuc(d).x = x2;
            Nuc(d).y = y2;
            
            if ~isempty(currentCell.RR)
                B = zeros(size(Ir));
                pixlist = find(currentCell.RR > 0);
                B(pixlist) = 1;
                BW = bwboundaries(B, 8, 'noholes');
                x1 = BW{1}(:,2);
                y1 = BW{1}(:,1);
                [x2, y2] = poly2cw(x1, y1);
                Whole(d).x = x2;
                Whole(d).y = y2;
            end
        end
    end

    I(:,:,1) = Ir; I(:,:,2) = Ig; I(:,:,3) = Ib;
    imshow(I); 

    % draw nucleus and soma
    for d = 1:length(CellsList{t})
        detect_ind = CellsList{t}(d);

        if tracks(detect_ind) ~= 0    
            
            currentCell = Cells(detect_ind);
            color = cols(tracks(detect_ind),:);
            switch mode
                case {6,9}
                    color = [0 0 0];
            end
            SomaFaceColor = color;
            SomaEdgeColor = color;
            NucleusColor = color;
            EndPointColor = [0 0 0];
            
            
            switch mode
                case 7
                     % patch for nucleus
                    NucleusColor = [1 0 0];
                    nucp = patch(Nuc(d).x, Nuc(d).y, 1, 'FaceColor', NucleusColor, 'FaceAlpha', 1, 'EdgeColor', NucleusColor, 'EdgeAlpha', 1, 'LineWidth', .5);
                case 12
                    NucleusColor = [1 0 0];
                    SomaFaceColor = [1 0 0];
                    SomaEdgeColor = [1 0 0];
                    % patch for soma interior
                    somap = patch(Soma(d).x, Soma(d).y, 1, 'FaceColor', SomaFaceColor, 'FaceAlpha', SomaFaceAlpha, 'EdgeColor', SomaEdgeColor, 'EdgeAlpha', SomaEdgeAlpha, 'LineWidth', EDGEWIDTH); %'none');

                    % patch for nucleus
                    nucp = patch(Nuc(d).x, Nuc(d).y, 1, 'FaceColor', NucleusColor, 'FaceAlpha', 1, 'EdgeColor', NucleusColor, 'EdgeAlpha', 1,'LineWidth', .5);

                otherwise
                    % patch for soma interior
                    somap = patch(Soma(d).x, Soma(d).y, 1, 'FaceColor', SomaFaceColor, 'FaceAlpha', SomaFaceAlpha, 'EdgeColor', SomaEdgeColor, 'EdgeAlpha', SomaEdgeAlpha, 'LineWidth', EDGEWIDTH); %'none');

                    % patch for nucleus
                    nucp = patch(Nuc(d).x, Nuc(d).y, 1, 'FaceColor', NucleusColor, 'FaceAlpha', 1, 'EdgeColor', NucleusColor, 'EdgeAlpha', 1,'LineWidth', .5);
            end
           
            
            % text cell ID
            switch mode
                case 0
                    cellID = currentCell.ID;
                    str = sprintf('%d',cellID);
                    text(max(Soma(d).x)+5, max(Soma(d).y)-5, str, 'Color', SomaEdgeColor);
                otherwise
            end
            
            % whole body segmentation and candidate endpoints
            if d < length(Whole)
                switch mode
                    case {4,5,10}
                        % patch for whole body
                        wholep = patch(Whole(d).x, Whole(d).y, ones(size(Whole(d).x)), 1, 'FaceColor', SomaFaceColor, 'FaceAlpha', WholeFaceAlpha, 'EdgeColor', SomaEdgeColor, 'EdgeAlpha', WholeEdgeAlpha,'LineWidth', EDGEWIDTH);
                        
                    otherwise
                end
            end
            
            % whole body segmentation and candidate endpoints
            if d < length(Whole)
            switch mode
                case {5,10,11}          
                    % candidate endpoints
                    for i = 1:size(currentCell.CandidateEndPoints,1)
                        r1 = currentCell.CandidateEndPoints(i,1);
                        c1 = currentCell.CandidateEndPoints(i,2);
                        s = 2;
                        rlist = [r1-s r1+s r1+s r1-s];
                        clist = [c1-s c1-s c1+s c1+s];
                        endp = patch(clist, rlist, 2* ones(size(clist)),1,  'FaceColor', EndPointColor , 'FaceAlpha', NucleusFaceAlpha, 'EdgeColor', EndPointColor, 'EdgeAlpha', NucleusEdgeAlpha);
                    end
                otherwise
            end
            end
            
        end
    end
    
    
    
    
    

    refresh;
    drawnow;
    pause(.1);

    
    set(f, 'Position', [1937 262 696 520]);
    F = getframe(gcf);
    I = F.cdata;
    Ir = I(:,:,1); Ig = I(:,:,2); Ib = I(:,:,3);
    
    
	% draw the neurites
    for d = 1:length(CellsList{t})
        detect_ind = CellsList{t}(d);

        if tracks(detect_ind) ~= 0    
            color = cols(tracks(detect_ind),:);    
            currentCell = Cells(detect_ind);

            NeuriteColor = round(color*255);
            switch mode
                case {5,6,9}
                    NeuriteColor = [0 0 0];
            end
            
            % color neurites or the cell body
            switch mode
                case {0,1,5,6,8,11}
                    FILMASK = currentCell.Neurites; 
                    Ir(FILMASK) = NeuriteColor(1);
                    Ig(FILMASK) = NeuriteColor(2);
                    Ib(FILMASK) = NeuriteColor(3);
                otherwise
            end
        end
    end
    I(:,:,1) = Ir; I(:,:,2) = Ig; I(:,:,3) = Ib;
    
    
    
    
    imshow(I);
	refresh;
    drawnow;
    pause(.1);
    
    
    % clip 7 pixels from each end
    Ir = I(:,:,1); Ig = I(:,:,2); Ib = I(:,:,3);
    Iout(:,:,1) = Ir(7:end-7, 7:end-7); 
    Iout(:,:,2) = Ig(7:end-7, 7:end-7); 
    Iout(:,:,3) = Ib(7:end-7, 7:end-7);
    
    % store the image for writing a movie file
    mv{t} = Iout;

end


