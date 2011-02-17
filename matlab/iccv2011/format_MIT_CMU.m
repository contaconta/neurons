

folderA = '/home/ksmith/data/faces/MIT_CMU/test/';
folderB = '/home/ksmith/data/faces/MIT_CMU/test-low/';
folderC = '/home/ksmith/data/faces/MIT_CMU/newtest/';

txtfile = '/home/ksmith/data/faces/MIT_CMU/list.txt';

imgfolder = '/home/ksmith/data/faces/MIT_CMU/faces/';

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

for i = 1:length(names)
    
    [pathstr name ext] = fileparts(names{i});
    
    % convert the gif to png
    pathstr = [pathstr '/'];
    cmd = ['convert ' pathstr name '.gif ' pathstr name '.png'];
    %disp(cmd);
    system(cmd);
    
    c = getfaces(A, [name '.png'], i);
    
    % display the image
    figure(1);
    newnames{i} = [pathstr name '.png'];
    I{i} = imread(newnames{i});
    imshow(I{i});

    hold on;
    
    if ~isempty(c)
        for k = 1:size(c,1)
            plot(c(k,1),c(k,2), 'go');
            
            x1 = c(k,1) - c(k,3)+1;
            y1 = c(k,2) - c(k,3)+1;
            x2 = c(k,1) + c(k,3)+1;
            y2 = c(k,2) + c(k,3)+1;
            
            h = line([x1 x2 x2 x1 x1], [y1 y1 y2 y2 y1]);
            set(h, 'Color', [0 1 0]);
        end
    end
    hold off;
        refresh;
    %pause(0.05);
    %pause;
    %close;
    
    
    % write the file
    filename = [imgfolder sprintf('%03d.png', i-1)];
    imwrite(I{i}, filename, 'PNG');
    
    
    
    fprintf(fid, '<frame>\n');
    fprintf(fid,'<PositionIndex>\n');
    fprintf(fid, '%d \n', i-1);
    fprintf(fid, '</PositionIndex>\n');
    
    
    
    if ~isempty(c)
        %str = [sprintf('%03d', i) ' ' ];
        str = [];
        
        for k = 1:size(c,1)
            w = 2*c(k,3) +1;
           
           	fprintf(fid,'<face>\n');
            fprintf(fid, '%f %f %f \n',c(k,1),c(k,2),w);
            fprintf(fid, '</face>\n');
            
        end 
        
        
        
    end
    
	fprintf(fid,'</frame>');
end


fprintf(fid, '</tagDocument>');
fclose(fid);  



% fclose(fid);
% 
% 
% fprintf(fid, '<?xml version=''1.0'' encoding=''utf-8''?> \n');
% fprintf(fid, '<tagDocument> \n');
% fprintf(fid, '<numberOfFrames>\n');
% fprintf(fid, '%d \n', NumFrames );
% fprintf(fid, '</numberOfFrames>\n');
% for i=1:NumFrames
%     fprintf(fid, '<frame>\n');
%     fprintf(fid,'<PositionIndex>\n');
%     fprintf(fid, '%d \n', i-1);
%     fprintf(fid, '</PositionIndex>\n');
%         for j=1:size(handles.annotations{i},1)
%             % todo find the mid point and subtract 1   floor (xmin + w / 2)
%             xmin = handles.annotations{i}(j,1);
%             ymin = handles.annotations{i}(j,2);
%             w = handles.annotations{i}(j,3);
%             x = floor(xmin + (w/2)) -1;
%             y = floor(ymin + (w/2)) -1;
%             fprintf(fid,'<neuron>\n');
%             fprintf(fid, '%f %f %f \n',x,y,w);
%             fprintf(fid, '</neuron>\n');
%         end
%     fprintf(fid,'</frame>');
% end
      



