function MASK = xml2mask(filename, arg2)
% XML2MASK
%
% MASK = xml2mask(filename, HOMEIMAGES)
% MASK = xml2maks(filename, [IMSIZE])
% Reads a labelme XML file, converts it to a binary mask (1 = object, 0= no
% object). The 2nd argument can be either the size of the image the
% annotation is describing, or the path of HOMEIMAGES containing the image
% the annotation is describing.
%
% Example:
%  filename = 'C:\atb\DATABASES\LabelMe\Annotations\05june05_static_indoor\p1010845.xml'
%  HOMEIMAGES = 'C:\atb\Databases\CSAILobjectsAndScenes\Images'
%  M = xml2mask(filename, HOMEIMAGES)
%  M = xml2mask(filename, [1536 1024])
%  imshow(M);
%
% Copyright © 2009 Kevin Smith
%
% See also 

% load annotation file:
v = loadXML(filename);

switch class(arg2)

    case 'string'
        % load image
        if nargin == 1
            %HOMEIMAGES = '\osshare\Work\Data\LabelMe\Images\'; %you can set here your default folder
            img = imread(fullfile(arg2, v.annotation.folder, v.annotation.filename));
            IMSIZE = size(img);
        end
    case 'double'
        IMSIZE = arg2;
    otherwise
        error('The second argument must be the image size (double) or path to HOMEIMAGES');
end


MASK = zeros(IMSIZE);
% draw each object (only non deleted ones)
Nobjects = length(v.annotation.object); n=0;
for i = 1:Nobjects

    if v.annotation.object(i).deleted == '0'
        n = n+1;
        %class{n} = strtrim(v.annotation.object(i).name); % get object name
        X = str2num(char({v.annotation.object(i).polygon.pt.x})); %get X polygon coordinates
        Y = str2num(char({v.annotation.object(i).polygon.pt.y})); %get Y polygon coordinates

        MTEMP = poly2mask(X,Y,IMSIZE(1), IMSIZE(2));
        %plot([X; X(1)],[Y; Y(1)], 'LineWidth', 4, 'color', [0 0 0]); hold on
        %h(n) = plot([X; X(1)],[Y; Y(1)], 'LineWidth', 2, 'color',colors(mod(n-1,6)+1)); hold on
    else
        MTEMP = zeros(IMSIZE);
    end

    MASK = MTEMP | MASK;
end
