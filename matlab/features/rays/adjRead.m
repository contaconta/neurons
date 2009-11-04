

adjFolder = '/osshare/DropBox/Dropbox/aurelien/superpixels/neighbors/';
d = dir([adjFolder '*.dat']);


for k = 1:length(d)
    
    fileRoot = regexp(d(k).name, '(\w*)[^\.]', 'match');
  	fileRoot = fileRoot{1};

    disp([' reading ' fileRoot]);
    fid = fopen([adjFolder fileRoot '.dat'], 'r');

    lmax = 31570;
    A = sparse([],[],[],lmax,lmax,0);

    while 1
        tline = fgetl(fid);
        if ~ischar(tline), break, end;
        %disp(tline);

        a = str2num(tline); %#ok<ST2NM>

        A(a(1),a(1)) = 1;
        for i = 2:length(a)
            A(a(1),a(i)) = 1;
        end
        %keyboard;
    end
    fclose(fid);

    A = max(A, A');
    
    save([adjFolder fileRoot '.mat'], 'A');
end