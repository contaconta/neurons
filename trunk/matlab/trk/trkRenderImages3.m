%function mv = trkRenderImages3(TMIN,TMAX,date_txt, num_txt,label_txt,cols,mv,Dlist,BLANK,FILAMENTS,Soma,tracks,D,N,DISPLAY_FIGURES) 
function mv = trkRenderImages3(TMIN,TMAX,R,cols,mv,DISPLAY_FIGURES)
% mv = trkRenderImages3(TMIN,TMAX,R,cols,mv,DISPLAY_FIGURES)
%
%   TMIN = first time step to render
%   TMAX = last time step to render
%   R = run data structure containing all data from the experimental run
%   cols = color map for drawing neurons
%   mv = the original image sequence to draw on top of
%   DISPLAY_FIGURES = flag to show the rendered figures




date_txt    = R.GlobalMeasures.Date;
num_txt     = R.GlobalMeasures.AssayPosition;
label_txt   = R.GlobalMeasures.Label;
Dlist       = R.Dlist;
BLANK       = zeros(size(mv{1}));
FILAMENTS   = R.FILAMENTS;
Soma        = R.Soma;
tracks      = R.tracks;
D           = R.D;

NPROCESSED = 0;
if isfield(R, 'N')
N           = R.N;
NPROCESSED = 1;
end

CONTRAST = 0.4;  %[0,1] 1 is normal contrast, 0 is VERY stretched contrast

B = zeros(size(mv{1},1), size(mv{1},2));

for t = TMIN:TMAX
    I = imadjust(mv{t}, [0; CONTRAST], [1; 0]);
    Ir = I(:,:,1); Ig = I(:,:,2); Ib = I(:,:,3);

    %% 1. draw the objects

    % draw nucleus and soma
    for i = 1:length(Dlist{t})
        d = Dlist{t}(i);
        
        if tracks(d) ~= 0
            if isfield(D, 'Happy')
                if D(d).Happy == 0
                    color = [.5 .5 .5];
                else
                    color = cols(tracks(d),:);
                    color = hsv2rgb(rgb2hsv(color) + [0 -.25 -.1]);
                end
            else
                color = cols(tracks(d),:);
                color = hsv2rgb(rgb2hsv(color) + [0 -.25 -.1]);
            end

            % color basic filament skeletons
            FILMASK = BLANK > Inf;
            FILMASK( FILAMENTS(d).PixelIdxList) = 1;
            FILMASK(Soma(d).PixelIdxList) = 0;
            Ir(FILMASK) = max(0, color(1) - .2);
            Ig(FILMASK) = max(0, color(2) - .2);
            Ib(FILMASK) = max(0, color(3) - .2);

            numNeurites = max(FILAMENTS(d).NeuriteID);
            filoIdxList = find(FILAMENTS(d).FilopodiaFlag);
            filoPixList = FILAMENTS(d).PixelIdxList(filoIdxList); %#ok<FNDSB>
            for j = 1:numNeurites
                if NPROCESSED
                    if D(d).Happy ~= 0
                        n = FILAMENTS(d).NIdxList(j);
                        nTrack = N(n).NeuriteTrack;
                        %MajorAxisLength = N(n).MajorAxisLength;
                        if nTrack == 0
                            neuritecolor = [.6 .6 .6];
                        else
                            neuritecolor = cols(nTrack,:);
                            %neuritecolor = getNeuriteColor(nTrack, color);
                            %neuritecolor = getNeuriteColor2(MajorAxisLength, color);
                        end
                    else
                        neuritecolor = [.6 .6 .6];
                    end
                else
                    neuritecolor = color;
                end
                
                
                neuritepixels = FILAMENTS(d).PixelIdxList( FILAMENTS(d).NeuriteID == j);
               [Ir Ig Ib] = colorHighlight(Ir,Ig,Ib,neuritepixels, neuritecolor);
               
%                 Ir(neuritepixels) = neuritecolor(1);
%                 Ig(neuritepixels) = neuritecolor(2);
%                 Ib(neuritepixels) = neuritecolor(3);
                Ir(neuritepixels) = color(1);
                Ig(neuritepixels) = color(2);
                Ib(neuritepixels) = color(3);
                
                %[Ir Ig Ib] = colorNeurites2(Ir,Ig,Ib,neuritepixels,filoPixList,neuritecolor);
            end
            
            % draw filopodia
%             filohsv = rgb2hsv(color) + [.085 0  .075];
%             filohsv(filohsv > 1) = filohsv(filohsv > 1) - 1;
%             filocolor = hsv2rgb(filohsv);
%             filocolor = color;
%             [Ir Ig Ib] = colorNeurites(d, Ir,Ig,Ib,FILAMENTS,filocolor);
            
            
%             % draw branch points
%             branchpts = FILAMENTS(d).PixelIdxList( FILAMENTS(d).NumKids >= 2);
%             branchcolor = hsv2rgb(rgb2hsv(color) + [0 0 .3]);
%             Ir(branchpts) = branchcolor(1);
%             Ig(branchpts) = branchcolor(2);
%             Ib(branchpts) = branchcolor(3);
            
            % color the soma
            SomaM = B > Inf;
            SomaM(Soma(d).PixelIdxList) = 1;
            SomaP = bwmorph(SomaM, 'remove');
            SomaP = bwmorph(SomaP, 'dilate');
            SomaP = bwmorph(SomaP, 'thin',1);
            somacolor = hsv2rgb(rgb2hsv(color) + [0 0 0]);
            Ir(SomaP) = somacolor(1);
            Ig(SomaP) = somacolor(2);
            Ib(SomaP) = somacolor(3);

            % color the nucleus
            Ir(D(d).PixelIdxList) = somacolor(1);
            Ig(D(d).PixelIdxList) = somacolor(2);
            Ib(D(d).PixelIdxList) = somacolor(3);
        end
    end

    I(:,:,1) = Ir; I(:,:,2) = Ig; I(:,:,3) = Ib;

    %% 2. render text annotations
    I = uint8(255*I);
    blk = [80 80 80];
    
    for i = 1:length(Dlist{t})
        d = Dlist{t}(i);

        % add text annotation
        if tracks(d) ~= 0
            if isfield(D, 'Happy')
                if D(d).Happy == 0
                    color = [.7 .7 .7];
                else
                    color = cols(tracks(d),:);
                    color = hsv2rgb(rgb2hsv(color) + [0 -.25 -.1]);
                end
            else
                color = cols(tracks(d),:);
                color = hsv2rgb(rgb2hsv(color) + [0 -.25 -.1]);
            end
            
            col = max(color, 0);
            rloc = max(1,D(d).Centroid(2) - 30);
            cloc = max(1,D(d).Centroid(1) + 20);
            I=trkRenderText(I,['id=' num2str(D(d).ID)], floor(255*col), [rloc, cloc], 'bnd2', 'left');
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
%             leafpts   = FILAMENTS(d).PixelIdxList( FILAMENTS(d).NumKids == 0);
%             Ir(leafpts) = 0;
%             Ig(leafpts) = 1;
%             Ib(leafpts) = 0;




function [Ir Ig Ib] = colorNeurites(dlist, Ir,Ig,Ib,FILAMENTS,color)

for n = dlist(:)'
    if ~isfield(FILAMENTS, 'FilopodiaFlag')
        [FILAMENTS neuritePixList] = trkFindFilopodia(n,FILAMENTS);
    else
        neuriteIdxList = find(FILAMENTS(n).FilopodiaFlag);
        neuritePixList = FILAMENTS(n).PixelIdxList(neuriteIdxList); %#ok<FNDSB>
    end
        
    Ir(neuritePixList) = color(1);
    Ig(neuritePixList) = color(2);
    Ib(neuritePixList) = color(3);
end


function [Ir Ig Ib] = colorNeurites2(Ir,Ig,Ib,NeuriteList,FiloList,color)


DrawList = intersect(NeuriteList, FiloList);

Ir(DrawList) = color(1);
Ig(DrawList) = color(2);
Ib(DrawList) = color(3);



% function ncolor = getNeuriteColor(Ntrack, color)
% 
% MODNUM = 8;
% 
% hoffset = (mod(Ntrack,MODNUM) + 1)/(MODNUM*2);
% if abs(hoffset) < .05
%     if sign(hoffset) == 1
%         hoffset = hoffset + .1;
%     else
%         hoffset = hoffset - .1;
%     end
% end
%     
%     
% hsign = double(mod(ceil(Ntrack/MODNUM),3) == 0);
% hsign(hsign == 0) = -1;
% hoffset = hoffset.*hsign;
% 
% hoffset
% 
% hcolor = rgb2hsv(color);
% hcolor = [hcolor(1) .66 hcolor(2)] + [0 hoffset 0];
% %hcolor(hcolor > 1) = hcolor(hcolor > 1) - 1;
% %hcolor(hcolor < 0) = hcolor(hcolor < 0) + 1;
% if hcolor(2) > 1
%     hcolor(2) = 1;
% end
% if hcolor(2) < .2
%     hcolor(2) = .2;
% end
% 
% 
% ncolor = hsv2rgb(hcolor);
% 
% % if mean(ncolor < .2)
% %     keyboard;
% % end
% % if isnan(ncolor)
% %     keyboard;
% % end
% 
% %ncolor = mean([ncolor; color]);



function [Ir Ig Ib] = colorHighlight(Ir,Ig,Ib,PixelIdxList, color)

H = size(Ir,1);
badinds = find(mod(PixelIdxList,H) == 0);
PixelIdxList(badinds) = [];  %#ok<FNDSB>

PixelIdxList = PixelIdxList + 1;


Ir(PixelIdxList) = color(1);
Ig(PixelIdxList) = color(2);
Ib(PixelIdxList) = color(3);


function ncolor = getNeuriteColor(Ntrack, color)

MODNUM = 8;

sval = (mod(Ntrack,MODNUM))/(MODNUM) + (1/MODNUM);
    
hcolor = rgb2hsv(color);
hcolor = [hcolor(1) sval hcolor(2)];


ncolor = hsv2rgb(hcolor);





function ncolor = getNeuriteColor2(val, color)

hcolor = rgb2hsv(color);

cl = (val - 20)/80;
s = cl;
s(s > 1) = 1;

ncolor = hsv2rgb([hcolor(1) s hcolor(3)]);



function I = colortips(I, N)

% CC.Connectivity = 8;
% CC.ImageSize = R.FILAMENTS(d).IMSIZE;
% CC.NumObjects = 1;
% CC.PixelIdxList = N.PixelIdxList;


