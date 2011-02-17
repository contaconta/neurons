clear; 

folderA = '/home/ksmith/data/faces/MIT_CMU/test/';
folderB = '/home/ksmith/data/faces/MIT_CMU/test-low/';
folderC = '/home/ksmith/data/faces/MIT_CMU/newtest/';

txtfile = '/home/ksmith/data/faces/MIT_CMU/list.txt';

imgfolder = '/home/ksmith/data/faces/MIT_CMU/faces/';

annfolder = '/home/ksmith/data/faces/MIT_CMU/';
annfile = '/home/ksmith/data/faces/MIT_CMU/faces.xml';
fid = fopen(annfile, 'w');


dA = dir([folderA '*.gif']);
dB = dir([folderB '*.gif']);
dC = dir([folderC '*.gif']);

nameA = {dA(:).name};
nameB = {dB(:).name};
nameC = {dC(:).name};

for i=1:length(nameA)
    nameA{i} = [folderA nameA{i}];
end
for i=1:length(nameB)
    nameB{i} = [folderB nameB{i}];
end
for i=1:length(nameC)
    nameC{i} = [folderC nameC{i}];
end

A = importdata(txtfile);

names = [nameA nameB nameC];

numFrames = length(names);

% xml stuff
fprintf(fid, '<?xml version=''1.0'' encoding=''utf-8''?> \n');
fprintf(fid, '<tagDocument> \n');
fprintf(fid, '<numberOfFrames>\n');
fprintf(fid, '%d \n', numFrames );
fprintf(fid, '</numberOfFrames>\n');

minwidth = Inf;

annotations = cell(1,length(names));
if exist([annfolder 'relabel.mat'], 'file');
    load([annfolder 'relabel.mat']);
end

for i = 1:length(names)
    
    [pathstr name ext] = fileparts(names{i});
    
    % convert the gif to png
    pathstr = [pathstr '/'];
    cmd = ['convert ' pathstr name '.gif ' pathstr name '.png'];
    system(cmd);
    
    
    
    % display the image
    figure(1);
    newnames{i} = [pathstr name '.png'];
    I{i} = imread(newnames{i});
    imshow(I{i});

    hold on;
    
    [c X Y] = getfaces(A, [name '.png'], i);
    
    if ~isempty(c)
        for k = 1:size(c,1)
            
%             plot(X(k,1),Y(k,1), 'g.');
%             plot(X(k,2),Y(k,2), 'g.');
%             plot(X(k,3),Y(k,3), 'g.');
%             plot(X(k,4),Y(k,4), 'g.');
%             plot(X(k,5),Y(k,5), 'g.');
%             plot(X(k,6),Y(k,6), 'g.');
            
            if ~isempty(annotations{i})
                xc = annotations{i}(k,1)+1;
                yc = annotations{i}(k,2)+1;
                w = annotations{i}(k,3);
                x1 = xc -w/2;
                y1 = yc - w/2;
                W = w;
                H = w;
            else
                x1 = X(k,3)-c(k,3)/2;
                y1 = Y(k,3)-c(k,3)/2;
                W = c(k,3);
                H = c(k,3);
            end
            h(k) = imrect(gca,  [ x1 y1 W H  ]);
            setFixedAspectRatioMode(h(k),1)
            
        end
    end
    hold off;
    refresh;
  
    disp('move the boxes to label the faces!');
    pause;
    %keyboard;
    
    
 
    
    fprintf(fid, '<frame>\n');
    fprintf(fid,'<PositionIndex>\n');
    fprintf(fid, '%d \n', i-1);
    fprintf(fid, '</PositionIndex>\n');
    
    
	if ~isempty(c)
        for k = 1:size(c,1)
            pos = getPosition(h(k)); %xmin ymin w h
            xc = round(pos(1) + pos(3)/2);
            yc = round(pos(2) + pos(3)/2);
            w = pos(3); %round(pos(3)/2);
            
            minwidth = min([minwidth w]);
            
            fprintf(fid,'<face>\n');
            fprintf(fid, '%f %f %f \n',xc-1,yc-1,w);
            fprintf(fid, '</face>\n');
            
         	annotations{i}(k,:) = [ xc,yc,w ]; %#ok<*SAGROW>
            
            x1 = xc - w/2;
            x2 = xc + w/2;
            y1 = yc - w/2;
            y2 = yc + w/2;
            l = line([x1 x2 x2 x1 x1], [y1 y1 y2 y2 y1]);
            
            %keyboard;
            
            delete(h(k));
        end
    end
    
    save([annfolder 'relabel.mat'], 'annotations');
    disp(['saved ' annfolder 'relabel.mat']);
    
    disp(['minimum width face is: [' num2str(minwidth) ' x ' num2str(minwidth) ']']);
          
    fprintf(fid,'</frame>');
    
    pause;
  	pause(0.05);
 
    close;
end

disp(['wrote ' annfile]);

fprintf(fid, '</tagDocument>');
fclose(fid);  