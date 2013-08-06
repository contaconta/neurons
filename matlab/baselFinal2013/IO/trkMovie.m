function trkMovie(mv, folder, resultsFolder, filename, rmFileFlag)
% folder  where you construct the movie
% resultsfolder  where the movie ends up

if nargin < 5
    rmFileFlag = 1;
end


disp('...writing temporary image files');
for i = 1:length(mv)
    imwrite(mv{i}, [folder sprintf('%03d',i) '.png'], 'PNG');
end
disp('...encoding movie');
oldpath = pwd;
cd(folder);

cmd_webm      = ['ffmpeg -v 0 -loglevel 0 -y -r 10 -i %03d.png -acodec libvorbis -b 2048k -y -s 696x520 -r 10 '  filename '.webm'];
cmd_mp4       = ['ffmpeg -v 0 -loglevel 0 -y -r 10 -i %03d.png  -vcodec libx264 -b 2048k -pix_fmt yuv420p -y -s 696x520 -r 10 ' filename '.mp4'];

system(cmd_mp4);
system(cmd_webm);

cmd_thumbnail = ['mv 050.png ' filename '.png'];
system(cmd_thumbnail);

A = imread([resultsFolder filename '.png']);
imwrite(A, [resultsFolder filename '.jpg']);

if rmFileFlag
	cmd = 'rm *.png'; 
    system(cmd);
end

cd(oldpath);

