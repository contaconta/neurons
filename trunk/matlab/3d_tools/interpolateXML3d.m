function interpolateXML3d(xmlfolder, dstfolder, search)
%INTERPOLATEXML3D interpolates missing slices in 3D volumes
%
%  interpolateXML3d(xmlfolder, dstfolder, search) given a folder containing
%  xml annotations XMLFOLDER and a search string SEARCH (eg '*_right.xml')
%  which will identify the correct files within the folder, determines
%  which slices are missing and interpolates to generate the missing
%  slices. The entire volume is output to DSTFOLDER as a set of binary
%  masks.
%
%  EXAMPLE (writes masks to ./masks/ for xml files with *_right.xml):
%  ------------------
%  interpolateXML3d('./annotations/', './masks/','*_right.xml') 
%


IMSIZE = [1536 1024];

SEARCHWINDOW = [35 35];  XYSEARCH = (SEARCHWINDOW-1)/2;
ZSEARCH = 4; %#ok<NASGU>

STEP = 4;

if nargin < 3
    search = '*_left.xml';
end
if nargin < 2
    dstfolder = '';
end

% first load the xmlfolder
d = dir([xmlfolder search]);




% loop through the images
for i = 1:length(d)-1

    %disp(['   loading ' d(i).name]);

    % parse the filename to find the z value (slice #)
    pat = '\d*';
    Za = str2double(regexp(d(i).name,pat, 'match'));
    %disp(num2str(Za));
    
    % get the mask for current Z
    filename = [xmlfolder d(i).name];
    Ma = xml2mask(filename, IMSIZE);
%     imshow(Ma);
%     pause(0.1);

    % next valid slice is i + 1
    % determine missing slices to be interpolated & search for next slice with valid contours
    
    Zb = str2double(regexp(d(i+1).name,pat, 'match')); %#ok<NASGU>
    filename = [xmlfolder d(i+1).name];
    Mb = xml2mask(filename, IMSIZE);
    
    % determine the missing Z masks
    MISSINGZ = Za+1:Zb-1;
    disp(['   interpolating slice ' num2str(MISSINGZ)]);
    
    Pa = bwboundaries(Ma);
    Pb = bwperim(Mb);
    %props = regionprops(bwboundaries(Ma), 'PixelIdxList');
    
    MASK = zeros(IMSIZE);
    
    % loop through objects in slice A
    for j = 1:length(Pa)
    
       pointlist = Pa{j};
       pr = pointlist(1:STEP:size(pointlist,1),1);
       pc = pointlist(1:STEP:size(pointlist,1),2);
       
       INTR = []; INTC = [];
       
       % loop through the pointlist, finding nearest point in Mb
       for k=1:length(pr)
            ra = pr(k);
            ca = pc(k);
            rmin = max(1,ra -XYSEARCH(1)); rmax = min(IMSIZE(1), ra + XYSEARCH(1));
            cmin = max(1,ca -XYSEARCH(2)); cmax = min(IMSIZE(2), ca + XYSEARCH(2));
           
            WINDOW = Pb( rmin:rmax, cmin:cmax);
            [rb cb] = find(WINDOW);
           
            if ~isempty(rb)
                %ind = dsearch(cb,rb,delaunay(cb,rb),ca,ra);
                dists = sqrt( (rb - repmat(ra,size(rb,1), size(rb,2))).^2 + (cb - repmat(ca, size(rb,1), size(rb,2)).^2));
                ind = find(dists == min(dists),1, 'first');
                
                int = [round((ra + rmin-1+rb(ind))/2) round((ca + cmin-1+cb(ind))/2)];
                %keyboard;
                INTR = [INTR; int(1)]; INTC = [INTC; int(2)]; %#ok<AGROW>
            end
            
            
       end
       
       %keyboard;
       %INTR = pr; INTC = pc;
        
       MTEMP = poly2mask(INTC,INTR,IMSIZE(1), IMSIZE(2));
       
       MASK = MTEMP | MASK;
    end
    
    % write the mask to the destination folder
    if ~exist(dstfolder, 'dir')
        mkdir(dstfolder);
    end
    
    filename1 = [dstfolder 'slice' number_into_string(Za, 10000) '.png'];
    imwrite(Ma, filename1, 'PNG');
    disp(['     wrote ' filename1]);
    filename2 = [dstfolder 'slice' number_into_string(MISSINGZ, 10000) '.png'];
    imwrite(MASK, filename2, 'PNG');
    disp(['     wrote ' filename2]);
    %keyboard;
    
    % valid contours must be within WINDOW distance

    % construct lines between contours

    % determine intersection with missing data planes

    % build contours on missing data planes, construct a binary mask

end


%% handle the i = length(d) case
i = length(d);
pat = '\d*';
Za = str2double(regexp(d(i).name,pat, 'match'));
filename = [xmlfolder d(i).name];
Ma = xml2mask(filename, IMSIZE);
filename1 = [dstfolder 'slice' number_into_string(Za, 10000) '.png'];
imwrite(Ma, filename1, 'PNG');
disp(['     wrote ' filename1]);
