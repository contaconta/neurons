function xml2mask(filename, HOMEIMAGES, output_path)
% REWRITE!!!
% Shows the image and polygons corresponding to an XML file.
% drawXML(filename, HOMEIMAGES)
%
% filename   = name of the XML annotation file (full path name)
% HOMEIMAGES = root folder that contains the images
%
% Example:
%  filename = 'C:\atb\DATABASES\LabelMe\Annotations\05june05_static_indoor\p1010845.xml'
%  HOMEIMAGES = 'C:\atb\Databases\CSAILobjectsAndScenes\Images'
%  drawXML(filename, HOMEIMAGES
%
% load annotation file:
v = loadXML(filename);

% load image
if nargin == 1
    HOMEIMAGES = 'C:\atb\Databases\CSAILobjectsAndScenes\Images'; %
you can set here your default folder
end

img = imread(fullfile(HOMEIMAGES, v.annotation.folder, v.annotation.filename));

MASK = zeros(size(img));
% draw each object (only non deleted ones)
Nobjects = length(v.annotation.object); n=0;
for i = 1:Nobjects

    if v.annotation.object(i).deleted == '0'
        n = n+1;
        class{n} = strtrim(v.annotation.object(i).name); % get object name
        X = str2num(char({v.annotation.object(i).polygon.pt.x})); %get X polygon coordinates
        Y = str2num(char({v.annotation.object(i).polygon.pt.y})); %get Y polygon coordinates

        MTEMP = poly2mask(X,Y,size(img,1), size(img,2));
        %plot([X; X(1)],[Y; Y(1)], 'LineWidth', 4, 'color', [0 0 0]); hold on
        %h(n) = plot([X; X(1)],[Y; Y(1)], 'LineWidth', 2, 'color',colors(mod(n-1,6)+1)); hold on
    else
        MTEMP = zeros(size(img));
    end

    MASK = MTEMP | MASK;
end

%figure; imshow(MASK);

h = figure('Visible', 'off');
imshow(MASK);
set(gca, 'Position', [0 0 1 1]);
[pathstr, name, ext] = fileparts(filename);
output_file = [output_path name '.png']
eval(['print -f' int2str(h) ' -dpng ' output_file]);
close(h);
