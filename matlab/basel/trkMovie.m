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

cmd_webm      = ['ffmpeg -r 10 -i %03d.png -acodec libvorbis -b 800k -y -s 696x520 -r 10 ' resultsFolder filename '.webm'];
system(cmd_webm);
cmd_mp4       = ['ffmpeg -r 10 -i %03d.png  -vcodec libx264 -b 800k  -y -s 696x520 -r 10 ' resultsFolder filename '.mp4'];
system(cmd_mp4);
cmd_thumbnail = ['mv 050.png ' resultsFolder filename '.png'];
system(cmd_thumbnail);
A = imread([resultsFolder filename '.png']);
imwrite(A, [resultsFolder filename '.jpg']);

cd(oldpath);

if rmFileFlag
	cmd = ['rm ' folder '*.png']; 
    system(cmd);
end
