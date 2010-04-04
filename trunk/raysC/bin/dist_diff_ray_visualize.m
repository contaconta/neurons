
im_dir = '/tmp/';
im_name = 'FIBSLICE0002';

img_name = {'/osshare/Work/Data/LabelMe/Images/FIBSLICE/FIBSLICE0002.png'};
g = imread(img_name{1});

ls_files=dir([im_dir im_name 'ray2_*.ppm']);

for i=1:size(ls_files,1)

    filename = ls_files(i).name;
    if(filename(1) == '.')
        continue;
    end

    ab_str = regexp(filename, '\d*', 'match');
    a = ab_str(3);
    b = ab_str(4);
    subplot(3,1,1);
    ray1 = [im_dir im_name 'ray1_' num2str(a{1}) '.ppm']
    im=read_32bitsimage(ray1,[size(g,2) size(g,1)]);
    imagesc(im);
    title(ray1);
    subplot(3,1,2);
    ray2 = [im_dir im_name 'ray1_' num2str(b{1}) '.ppm']
    im=read_32bitsimage(ray2,[size(g,2) size(g,1)]);
    imagesc(im);
    title(ray2);
    subplot(3,1,3);
    im=read_floatimage([im_dir filename],[size(g,2) size(g,1)]);
    imagesc(im);
    title(filename);
    refresh;
    pause(0.1);

    %pause

    % you can interrupt by pressing Ctrl-C
    % Use the "return" or "dbcont" command to resume execution of the loop.
    % Careful : this will make the script run slower
    % dbloop
end
