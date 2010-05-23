function name= hostname()
%
% This function reports the name of computer on which Matlab is currently
% running.
%
% Dr. Phillip M. Feldman, 13 May 2009
%
% The builtin string variable <computer> contains the string 'PCWIN' if we
% are running in a 32-bit Windows environment, and 'MACI' or 'MAC' if we
% are running on a Mac.  Otherwise, the presumption is that we are running
% under either Linux or Solaris.  In all of these environments, Matlab's
% unix() function sends a command to the shell and returns the output.  The
% specific command that must be sent to the shell depends on the
% environment in which we are running.
%
if strcmp(computer,'PCWIN') | strcmp(computer,'MACI') | strcmp(computer,'MAC')
   [status,name]= unix('hostname');
else
   [status,name]= unix('/bin/uname -n');
end

% Strip any trailing blanks:
name= deblank(name);