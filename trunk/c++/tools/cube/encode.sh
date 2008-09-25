#!/bin/bash

files=mf://$1/*.png
video=$1/output.avi

mencoder -ovc lavc -lavcopts vcodec=msmpeg4v2:vpass=1:"vbitrate=15000:mbd=2:keyint=132:vqblur=1.0:cmp=2:subcmp=2:dia=2:mv0:last_pred=3" -mf type=png:fps=11 -nosound -o /dev/null $files
mencoder -ovc lavc -lavcopts vcodec=msmpeg4v2:vpass=2:"vbitrate=15000:mbd=2:keyint=132:vqblur=1.0:cmp=2:subcmp=2:dia=2:mv0:last_pred=3" -mf type=png:fps=11 -nosound -o $video $files

# mplayer -loop 0 $video
