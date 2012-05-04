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


%BITRATE = 15000000;
%BITRATE = 6000000;
BITRATE = 4000000;
%cmd1 = ['mencoder -ovc lavc -lavcopts vcodec=msmpeg4v2:vpass=1:"vbitrate=' num2str(BITRATE) ':mbd=2:keyint=132:vqblur=1.0:cmp=2:subcmp=2:dia=2:mv0:last_pred=3" -mf type=png:fps=10 -nosound -o /dev/null mf://*.png -really-quiet'];
%cmd2 = ['mencoder -ovc lavc -lavcopts vcodec=msmpeg4v2:vpass=2:"vbitrate=' num2str(BITRATE) ':mbd=2:keyint=132:vqblur=1.0:cmp=2:subcmp=2:dia=2:mv0:last_pred=3" -mf type=png:fps=10 -nosound -o ' resultsFolder filename ' mf://*.png -really-quiet'];

%-ovc x264 -x264encopts bitrate=3000 pass=1 nr=2000

%cmd1 = ['mencoder -ovc x264 -x264encopts bitrate=' num2str(BITRATE) 'pass=1 nr=2000 -mf type=png:fps=10 -nosound -o /dev/null mf://*.png -really-quiet'];
%cmd2 = ['mencoder -ovc x264 -x264encopts bitrate=' num2str(BITRATE) 'pass=2 nr=2000 -mf type=png:fps=10 -nosound -o -nosound -o ' resultsFolder filename ' mf://*.png -really-quiet'];
% use ffmpeg, it's much better
%keyboard;
%cmd_vorbis    = ['ffmpeg -r 10 -i %03d.png -acodec libvorbis -b 600k -y -s 696x520 -r 10 ' resultsFolder filename '.ogv'];
%system(cmd_vorbis);
cmd_webm      = ['ffmpeg -r 10 -i %03d.png -acodec libvorbis -b 800k -y -s 696x520 -r 10 ' resultsFolder filename '.webm'];
system(cmd_webm);
cmd_mp4       = ['ffmpeg -r 10 -i %03d.png  -vcodec libx264 -b 800k  -y -s 696x520 -r 10 ' resultsFolder filename '.mp4'];
system(cmd_mp4);
A = imread([resultsFolder filename '.png']);
imwrite(A, [resultsFolder filename '.jpg']);

cd(oldpath);



if rmFileFlag
	cmd = ['rm ' folder '*.png']; 
    system(cmd);
end





%BITRATE = '5000';
%BITRATE = '25000000';
%FPS = '10';
%cmd = ['mencoder "mf://*.png" -mf fps=' FPS ' -o ' resultsFolder filename ' -ovc xvid -xvidencopts bitrate=' BITRATE ' -really-quiet'];
% cmd = ['ffmpeg -r 10 -b 600k -i %03d.png ' resultsFolder filename];
%disp(cmd);
%system(cmd);


% vidObj = VideoWriter(filename);
% vidObj.FrameRate = 10;
% vidObj.Quality = 90;
% open(vidObj);
% 
% for t = 1:length(mv)
% 
% 	M(t) =  im2frame(mv{t});
% 
% 	writeVideo(vidObj,M(t));
% 
% end
% 
% 
% 
% close(vidObj);
