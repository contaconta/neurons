function S = sectohhmmss(varargin)
%SECTOHHMMSS Converts number seconds (s) into 'HH:MM:SS.XX' format
%
%   S = sectohhmmss(s, 'short') takes a number of seconds s (output from toc, for
%   example) and converts it to a human-readable format 'HH:MM:SS.XX' as a
%   string S.  Optional 'short' flag removes milliseconds from the string.
%
%   example:  sectohhmmss(5692.34) gives '1:34:52.34'
%
%   Copyright 2008 Kevin Smith
%
%   See also TIC, TOC, CPUTIME

if size(varargin) == 1
    s = varargin{1};
    short = 0;
else
    s = varargin{1};
    short = varargin{2};
end


if ischar(s)
    s = str2double(s);
end

HH = floor(s/3600);
MM = floor((s - HH*3600)/60);
SS = (s - HH*3600 - MM*60);

if strcmp(short, 'short')
    SS = floor(SS);
end

S = [ number_into_string(HH,99) ':' number_into_string(MM,99) ':' number_into_string(SS,99) ];