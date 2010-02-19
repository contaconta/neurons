function polys2xml(polys, imfile, xmlfile, imfolder, xmlfolder, label, IMSIZE)



v.annotation.filename = imfile;
v.annotation.folder = imfolder;
v.annotation.source.sourceImage = 'Source Image';
v.annotation.source.sourceAnnotation = 'Annotator3D tool';




for p = 1:length(polys)

    v.annotation.object(p).name = label;
    v.annotation.object(p).deleted = 0;
    v.annotation.object(p).verified = 0;
    v.annotation.object(p).date = datestr(now, 'dd-mmm-yyyy HH:MM:SS');
    v.annotation.object(p).id = num2str(p -1);
    v.annotation.object(p).polygon.username = 'smith';
    
    pol = polys{p};

    for n = 1:size(pol,1)
        v.annotation.object(p).polygon.pt(n).x = num2str(pol(n,1));
        v.annotation.object(p).polygon.pt(n).y = num2str(pol(n,2));
    end
end

v.annotation.imagesize.nrows = num2str(IMSIZE(1));
v.annotation.imagesize.nrows = num2str(IMSIZE(2));

xml = struct2xml(v);

destinationfile = [xmlfolder xmlfile];

% Open file
fid = fopen(destinationfile,'w');
fprintf(fid, xml);
% Close file
fclose(fid);

disp(['wrote ' destinationfile]);

 %   keyboard;