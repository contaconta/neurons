


postrain = '/home/ksmith/data/persons/INRIAPerson/96X160H96/Train/pos/';
negtrain = '/home/ksmith/data/persons/INRIAPerson/Train/neg/';
testfolder = '/home/ksmith/data/persons/INRIAPerson/Test/pos/';
annfolder = '/home/ksmith/data/persons/INRIAPerson/Test/annotations/';
testdest = '/home/ksmith/data/persons/INRIA/test/images/';
testroot = '/home/ksmith/data/persons/INRIA/test/';

dest1 = '/home/ksmith/data/persons/INRIA/train/pos/';
dest2 = '/home/ksmith/data/persons/INRIA/train/neg/';


% d = dir([postrain '*.png']);
% 
% for i = 1:length(d)
%     
%     file1 = [postrain d(i).name];
%     I = imread(file1);
%     I = rgb2gray(I);
%     I = I(33:33+96-1,:);
%     imshow(I);
%     drawnow;
%     
%     file2 = [dest1 'person' sprintf('%05d', i) '.png'];
%     imwrite(I, file2, 'PNG');
%     disp(['writing ' file2]);
% end
% 
% 
% d1 = dir([negtrain '*.png']);
% d2 = dir([negtrain '*.jpg']);
% 
% d = [d1; d2];
% 
% for i = 1:length(d)
%     file1 = [negtrain d(i).name];
%     I = imread(file1);
%     I = rgb2gray(I);
%     imshow(I);
%     drawnow;
%     
%     file2 = [dest2 'nonperson' sprintf('%05d', i) '.png'];
%     imwrite(I, file2, 'PNG');
%     disp(['writing ' file2]);
% end

d = dir([testfolder '*.png']);
numFrames = length(d);
minw = Inf; minh = Inf;

annfile = [testroot 'persons.xml'];
fid2 = fopen(annfile, 'w');

% xml stuff
fprintf(fid2, '<?xml version=''1.0'' encoding=''utf-8''?> \n');
fprintf(fid2, '<tagDocument> \n');
fprintf(fid2, '<numberOfFrames>\n');
fprintf(fid2, '%d \n', numFrames );
fprintf(fid2, '</numberOfFrames>\n');


for i = 1:length(d)

    fprintf(fid2, '<frame>\n');
    fprintf(fid2,'<PositionIndex>\n');
    fprintf(fid2, '%d \n', i-1);
    fprintf(fid2, '</PositionIndex>\n');
    
    iname = [testfolder d(i).name];
    aname = [annfolder d(i).name];
    [pth nm ext] = fileparts(aname);
    aname = [pth '/' nm '.txt'];
        
    figure(1);     clf; cla; hold on;
    I = imread(iname);
    I = rgb2gray(I);
    imshow(I);
    testname = [testdest sprintf('%03d', i) '.png'];
    imwrite(I, testname, 'PNG');
    disp(['wrote ' testname]);
    
    fid=fopen(aname);
    while 1
        tline = fgetl(fid);
        if ~ischar(tline), break, end
        %disp(tline)
        
        if length(tline > 22) %#ok<*ISMT>
            if strcmp(tline(1:23), 'Bounding box for object')
                m = regexp(tline, '\d*', 'match');
                x1 = str2double(m(2));
                y1 = str2double(m(3));
                x2 = str2double(m(4));
                y2 = str2double(m(5));

                w = x2 - x1 + 1;
                h = y2 - y1 + 1;
                minw = min([minw w]);
                minh = min([minh h]);
                
                h1 = line([x1 x2 x2 x1 x1], [y1 y1 y2 y2 y1]);
                set(h1, 'Color', [0 1 0]);
                %disp('got here!');
                
                fprintf(fid2,'<person>\n');
                fprintf(fid2, '%f %f %f %f\n',x1,x2,y1,y2);
                fprintf(fid2, '</person>\n');
            end
        end
    end
    fclose(fid);
    
    disp(['[minw  minh] = [' num2str(minw) '  ' num2str(minh) ']' ]);
    

 	drawnow;
    %pause;

    hold off;
    %close;
    
    fprintf(fid2,'</frame>');
end
    
fprintf(fid2, '</tagDocument>');
fclose(fid2);  

    