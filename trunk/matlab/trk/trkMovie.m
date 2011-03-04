function trkMovie(mv, folder, resultsFolder, filename)




disp('...writing temporary image files');
for i = 1:length(mv)
    imwrite(mv{i}, [folder sprintf('%03d',i) '.png'], 'PNG');
end
disp('...encoding movie');
oldpath = pwd;
cd(folder);
%BITRATE = '5000';
%BITRATE = '25000000';
%FPS = '10';
%cmd = ['mencoder "mf://*.png" -mf fps=' FPS ' -o ' resultsFolder filename ' -ovc xvid -xvidencopts bitrate=' BITRATE ' -really-quiet'];
% cmd = ['ffmpeg -r 10 -b 600k -i %03d.png ' resultsFolder filename];
%disp(cmd);
%system(cmd);

BITRATE = 15000000;
cmd1 = ['mencoder -ovc lavc -lavcopts vcodec=msmpeg4v2:vpass=1:"vbitrate=' num2str(BITRATE) ':mbd=2:keyint=132:vqblur=1.0:cmp=2:subcmp=2:dia=2:mv0:last_pred=3" -mf type=png:fps=10 -nosound -o /dev/null mf://*.png -really-quiet'];
cmd2 = ['mencoder -ovc lavc -lavcopts vcodec=msmpeg4v2:vpass=2:"vbitrate=' num2str(BITRATE) ':mbd=2:keyint=132:vqblur=1.0:cmp=2:subcmp=2:dia=2:mv0:last_pred=3" -mf type=png:fps=10 -nosound -o ' resultsFolder filename ' mf://*.png -really-quiet'];

system(cmd1);
system(cmd2);


cd(oldpath);

cmd = ['rm ' folder '*.png'];
%disp(cmd);
system(cmd);








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