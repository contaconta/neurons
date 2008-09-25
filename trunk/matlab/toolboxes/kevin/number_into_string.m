function  nstr = number_into_string(num, max_num)
%NUMBERINTOSTRING Converts a number into a string with leading 0's.
%
%   nstr = number_into_string(num, max_num) takes a number as input and 
%   returns a formatted string containing the number.  Adds leading 0's to 
%   the front of the number for file-writing and formatting purposes.  The 
%   number of leading zeros is determined by 'max_num'.
%
%   example:  number_into_string(56,10000) would return '00056'.
%   Copyright 2008 Kevin Smith
%
%   See also DISP, SECTOHHMMSS, NUM2STR.

if num < 1
    max_order = floor(log10(max_num));
    n_order = 0;
else
    max_order = floor(log10(max_num));
    n_order = floor(log10(num));
end
    
nstr = num2str(num);

if (max_order > n_order) && (n_order >= 0)
    for i = 1:max_order - n_order
        nstr = strcat('0', nstr);
    end
end
