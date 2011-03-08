%function mv = trkRenderImages3(TMIN,TMAX,date_txt, num_txt,label_txt,cols,mv,Dlist,BLANK,FILAMENTS,Soma,tracks,D,N,DISPLAY_FIGURES) 
function mv = trkRenderImages4(TMIN,TMAX,R,cols,mv,DISPLAY_FIGURES)
% mv = trkRenderImages3(TMIN,TMAX,R,cols,mv,DISPLAY_FIGURES)
%
%   TMIN = first time step to render
%   TMAX = last time step to render
%   R = run data structure containing all data from the experimental run
%   cols = color map for drawing neurons
%   mv = the original image sequence to draw on top of
%   DISPLAY_FIGURES = flag to show the rendered figures


warning off all

date_txt    = R.GlobalMeasures.Date;
num_txt     = R.GlobalMeasures.AssayPosition;
label_txt   = R.GlobalMeasures.Label;
Dlist       = R.Dlist;
BLANK       = zeros([size(mv{1},1) size(mv{1},2)]);
FILAMENTS   = R.FILAMENTS;
Soma        = R.Soma;
tracks      = R.tracks;
D           = R.D;

if isfield(R, 'N')
    N = R.N;
else
    error(' You need to do post-processing!');
end

CONTRAST = 0.4;  %[0,1] 1 is normal contrast, 0 is VERY stretched contrast



for t = TMIN:TMAX
    if mod(t,10) == 0
        disp(['...rendering frame ' num2str(t)]);
    end
    I = imadjust(mv{t}, [0; CONTRAST], [1; 0]);
    Ir = I(:,:,1); Ig = I(:,:,2); Ib = I(:,:,3);

    %% 1. draw the objects
    
    I0 = drawI0(t,cols,Ir,Ig,Ib,date_txt,num_txt,label_txt,Dlist,BLANK,FILAMENTS,Soma,tracks,D,N);
    I1 = drawI1(t,cols,Ir,Ig,Ib,date_txt,num_txt,label_txt,Dlist,BLANK,FILAMENTS,Soma,tracks,D,N);
    I2 = drawI2(t,cols,Ir,Ig,Ib,date_txt,num_txt,label_txt,Dlist,BLANK,FILAMENTS,Soma,tracks,D,N);
    I3 = drawI3(t,cols,Ir,Ig,Ib,date_txt,num_txt,label_txt,Dlist,BLANK,FILAMENTS,Soma,tracks,D,N);
    %I4 = drawI4(t,cols,Ir,Ig,Ib,date_txt,num_txt,label_txt,Dlist,BLANK,FILAMENTS,Soma,tracks,D,N);

   
    
    Ibig = zeros([2*size(I,1),2*size(I,2),3], 'uint8');
    Ibig(1:size(I,1),1:size(I,2),:) = I0;
    Ibig(1:size(I,1),size(I,2)+1:2*size(I,2),:) = I1;
    Ibig(size(I,1)+1:2*size(I,1),1:size(I,2),:) = I2;
    Ibig(size(I,1)+1:2*size(I,1),size(I,2)+1:2*size(I,2),:) = I3;
    Ibig(:,size(I,2),:) = 80;
    Ibig(size(I,1),:,:) = 80;
    
    
    % show the image
    if DISPLAY_FIGURES
        imshow(Ibig);
        drawnow;
    end

    % store the image for writing a movie file
    mv{t} = Ibig;
    
    
end


warning on all



    
    
function I = drawI0(t,cols,Ir,Ig,Ib,date_txt,num_txt,label_txt,Dlist,BLANK,FILAMENTS,Soma,tracks,D,N)
I(:,:,1) = Ir; I(:,:,2) = Ig; I(:,:,3) = Ib;
I = uint8(255*I);
blk = [80 80 80];
I=trkRenderText(I,date_txt, blk, [10, 20], 'bnd2', 'left');
I=trkRenderText(I,num_txt, blk, [10, 175], 'bnd2', 'left');
I=trkRenderText(I,label_txt, blk, [10, 240], 'bnd2', 'left');
I=trkRenderText(I,'Original', blk, [10, 460], 'bnd2', 'left');

function I = drawI1(t,cols,Ir,Ig,Ib,date_txt,num_txt,label_txt,Dlist,BLANK,FILAMENTS,Soma,tracks,D,N)

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
        for j = 1:numNeurites
            if D(d).Happy ~= 0
                n = FILAMENTS(d).NIdxList(j);
                nTrack = N(n).NeuriteTrack;
                if nTrack == 0
                    neuritecolor = [.6 .6 .6];
                else
                    neuritecolor = cols(nTrack,:);
                end
            else
                neuritecolor = [.6 .6 .6];
            end

           % color the neurites 
           neuritepixels = FILAMENTS(d).PixelIdxList( FILAMENTS(d).NeuriteID == j);
           [Ir Ig Ib] = colorHighlight(Ir,Ig,Ib,neuritepixels, color);

%                 Ir(neuritepixels) = neuritecolor(1);
%                 Ig(neuritepixels) = neuritecolor(2);
%                 Ib(neuritepixels) = neuritecolor(3);
            Ir(neuritepixels) = color(1);
            Ig(neuritepixels) = color(2);
            Ib(neuritepixels) = color(3);

            %[Ir Ig Ib] = colorFilopodia(Ir,Ig,Ib,neuritepixels,filoPixList,neuritecolor);
        end


        % color the soma
        SomaM = BLANK > Inf;
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
I=trkRenderText(I,'Neuron_Tracking', blk, [10, 460], 'bnd2', 'left');






function I = drawI2(t,cols,Ir,Ig,Ib,date_txt,num_txt,label_txt,Dlist,BLANK,FILAMENTS,Soma,tracks,D,N)

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
        for j = 1:numNeurites
            if D(d).Happy ~= 0
                n = FILAMENTS(d).NIdxList(j);
                nTrack = N(n).NeuriteTrack;
                if nTrack == 0
                    neuritecolor = [.6 .6 .6];
                else
                    neuritecolor = cols(nTrack,:);                    
                end
            else
                neuritecolor = [.6 .6 .6];
            end

           % color the neurites 
           neuritepixels = FILAMENTS(d).PixelIdxList( FILAMENTS(d).NeuriteID == j);
           [Ir Ig Ib] = colorHighlight(Ir,Ig,Ib,neuritepixels, neuritecolor);

            Ir(neuritepixels) = neuritecolor(1);
            Ig(neuritepixels) = neuritecolor(2);
            Ib(neuritepixels) = neuritecolor(3);
            
%             filoColor = [0 .8 0];
%             filoIdxList = find(FILAMENTS(d).FilopodiaFlag);
%             filoPixList = FILAMENTS(d).PixelIdxList(filoIdxList); %#ok<FNDSB>
%             [Ir Ig Ib] = colorFilopodia(Ir,Ig,Ib,neuritepixels,filoPixList,filoColor);
        end
        
        


        % color the soma
        SomaM = BLANK > Inf;
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
I=trkRenderText(I,'Neurite_Tracking', blk, [10, 460], 'bnd2', 'left');

function I = drawI3(t,cols,Ir,Ig,Ib,date_txt,num_txt,label_txt,Dlist,BLANK,FILAMENTS,Soma,tracks,D,N)

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
                color = [0.2235    0.2235    0.9020];
            end
        else
            color = cols(tracks(d),:);
            color = hsv2rgb(rgb2hsv(color) + [0 -.25 -.1]);
            color = [0.2235    0.2235    0.9020];
        end

        % color basic filament skeletons
        FILMASK = BLANK > Inf;
        FILMASK( FILAMENTS(d).PixelIdxList) = 1;
        FILMASK(Soma(d).PixelIdxList) = 0;
        Ir(FILMASK) = max(0, color(1) - .2);
        Ig(FILMASK) = max(0, color(2) - .2);
        Ib(FILMASK) = max(0, color(3) - .2);

        numNeurites = max(FILAMENTS(d).NeuriteID);
        for j = 1:numNeurites
            if D(d).Happy ~= 0
                n = FILAMENTS(d).NIdxList(j);
                nTrack = N(n).NeuriteTrack;
                if nTrack == 0
                    neuritecolor = [.6 .6 .6];
                else
                    %neuritecolor = cols(nTrack,:);
                    neuritecolor = color;
                end
            else
                neuritecolor = [.6 .6 .6];
            end

           % color the neurites 
           neuritepixels = FILAMENTS(d).PixelIdxList( FILAMENTS(d).NeuriteID == j);
           [Ir Ig Ib] = colorHighlight(Ir,Ig,Ib,neuritepixels, color);

            Ir(neuritepixels) = neuritecolor(1);
            Ig(neuritepixels) = neuritecolor(2);
            Ib(neuritepixels) = neuritecolor(3);

            if D(d).Happy ~= 0
                filoColor = [.9 .1 .1];
                filoIdxList = find(FILAMENTS(d).FilopodiaFlag);
                filoPixList = FILAMENTS(d).PixelIdxList(filoIdxList); %#ok<FNDSB>
                [Ir Ig Ib] = colorFilopodia(Ir,Ig,Ib,neuritepixels,filoPixList,filoColor);
            end
            
        end
        

        % color the soma
        SomaM = BLANK > Inf;
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
                color = [0.2235    0.2235    0.9020];
            end
        else
            color = cols(tracks(d),:);
            color = hsv2rgb(rgb2hsv(color) + [0 -.25 -.1]);
            color = [0.2235    0.2235    0.9020];
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
I=trkRenderText(I,'Filopodia', blk, [10, 460], 'bnd2', 'left');


function I = drawI4(t,cols,Ir,Ig,Ib,date_txt,num_txt,label_txt,Dlist,BLANK,FILAMENTS,Soma,tracks,D,N)

% draw nucleus and soma
for i = 1:length(Dlist{t})
    d = Dlist{t}(i);

    if tracks(d) ~= 0
        if isfield(D, 'Happy')
            if D(d).Happy == 0
                color = [.5 .5 .5];
            else
%                 color = cols(tracks(d),:);
%                 color = hsv2rgb(rgb2hsv(color) + [0 -.25 -.1]);
                color = [0.2235    0.2235    0.9020];
            end
        else
%             color = cols(tracks(d),:);
%             color = hsv2rgb(rgb2hsv(color) + [0 -.25 -.1]);
            color = [0.2235    0.2235    0.9020];
        end

        % color basic filament skeletons
        FILMASK = BLANK > Inf;
        FILMASK( FILAMENTS(d).PixelIdxList) = 1;
        FILMASK(Soma(d).PixelIdxList) = 0;
        Ir(FILMASK) = max(0, color(1) - .2);
        Ig(FILMASK) = max(0, color(2) - .2);
        Ib(FILMASK) = max(0, color(3) - .2);

        numNeurites = max(FILAMENTS(d).NeuriteID);
        for j = 1:numNeurites
            if D(d).Happy ~= 0
                n = FILAMENTS(d).NIdxList(j);
                nTrack = N(n).NeuriteTrack;
                if nTrack == 0
                    neuritecolor = color;
                else
                    expandcontract = D(d).KevinTotalCableLengthExpand;
                    %expandcontract = N(n).Expand;
                    neuritecolor = getNeuriteColor(expandcontract, color);
                end
            else
                neuritecolor = [.6 .6 .6];
            end

           % color the neurites 
           neuritepixels = FILAMENTS(d).PixelIdxList( FILAMENTS(d).NeuriteID == j);
           [Ir Ig Ib] = colorHighlight(Ir,Ig,Ib,neuritepixels, neuritecolor);

            Ir(neuritepixels) = neuritecolor(1);
            Ig(neuritepixels) = neuritecolor(2);
            Ib(neuritepixels) = neuritecolor(3);
          
            %[Ir Ig Ib] = colorFilopodia(Ir,Ig,Ib,neuritepixels,filoPixList,neuritecolor);
        end


        % color the soma
        SomaM = BLANK > Inf;
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
%                 color = cols(tracks(d),:);
%                 color = hsv2rgb(rgb2hsv(color) + [0 -.25 -.1]);
                color = [0.2235    0.2235    0.9020];
            end
        else
%             color = cols(tracks(d),:);
%             color = hsv2rgb(rgb2hsv(color) + [0 -.25 -.1]);
            color = [0.2235    0.2235    0.9020];
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



















function [Ir Ig Ib] = colorFilopodia(Ir,Ig,Ib,NeuriteList,FiloList,color)


DrawList = intersect(NeuriteList, FiloList);

Ir(DrawList) = color(1);
Ig(DrawList) = color(2);
Ib(DrawList) = color(3);

[Ir Ig Ib] = colorHighlight(Ir,Ig,Ib,DrawList, color);


function [Ir Ig Ib] = colorHighlight(Ir,Ig,Ib,PixelIdxList, color)

H = size(Ir,1);
badinds = find(mod(PixelIdxList,H) == 0);
PixelIdxList(badinds) = [];  %#ok<FNDSB>

PixelIdxList = PixelIdxList + 1;


Ir(PixelIdxList) = color(1);
Ig(PixelIdxList) = color(2);
Ib(PixelIdxList) = color(3);




function ncolor = getNeuriteColor(val, color)

hcolor = rgb2hsv(color);

if val == 1
    s = 1;
    ncolor = [0 1 0];
elseif val == 0
    ncolor = color;
else
    s = hcolor(2) / 2;
    ncolor = [1 0 0];
end


%ncolor = hsv2rgb([hcolor(1) s hcolor(3)]);


