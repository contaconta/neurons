function W = wristwatch(varargin)
%WRISTWATCH is a utility to help you time your code and make an ETA for loops.
%
%   W = wristwatch(W,...) is a utility to keep track of time in iterative 
%   loops in your code.   Wristwatch provides an update on the elapsed 
%   time and remaining time whenever a defined number of iterations are 
%   passed. Initialize the wristwatch by including the argument 'start', 
%   followed by other optional arguments: 
%   'end'       - the maximum number of iterations in the loop
%   'every'     - how often (in iterations) wristwatch should update
%   'text'      - text to pass to the iteration update
%   Calling wristwatch(W, 'update', n) within the loop will trigger a
%   status update every X iterations defined by 'every'.
%
%   Example:
%   ---------------------
%   W = wristwatch('start', 'end', 1000, 'every', 100);
%   for n = 1:1000
%       pause(.01);
%       W = wristwatch(W, 'update', n, 'text', '...how long will it take? ');
%   end
%
%   Copyright © 2008 Kevin Smith
%   See also TIC, TOC, CPUTIME, TIMER, SECTOHHMMSS
UPDATE = 0;

for i=1:nargin  
    if isa(varargin{i} , 'struct'); W = varargin{1}; end    

    if strcmp(varargin{i}, 'start')
        tic;
        W.N = 1;
        W.n = 1;
        W.time_per_iter = 0;
        W.time_elapsed = 0;
        W.step = 1;   
        W.text = '';
        W.remain = 0;
        W.display = 1;
        UPDATE = 0;
    end      
    if strcmp(varargin{i}, 'end')
        W.N = varargin{i+1};
    end
    if strcmp(varargin{i}, 'update')
        UPDATE = 1;
        W.n = varargin{i+1};  
    end
    if strcmp(varargin{i}, 'every')
        W.step = varargin{i+1};
    end
    if strcmp(varargin{i}, 'text')
        W.text = varargin{i+1};
    end    
    if strcmp(varargin{i}, 'display')
        W.display = varargin{i+1};
    end
end

 
if (UPDATE == 1) && ((mod(W.n, W.step) == 0) || (W.n == W.N))
    W.time_elapsed = toc;
    W.time_per_iter = W.time_elapsed/W.n;
    W.remain  = (W.N - W.n)*W.time_per_iter;   
    if W.display
        disp([ W.text number_into_string(W.n, W.N) '/' num2str(W.N) ' [elapsed time ' sectohhmmss(W.time_elapsed, 'short')  ' | ' sectohhmmss(W.remain, 'short') ' remaining]']);
    end
end

if (UPDATE == 1) && (W.n == 1)
    if W.display
        disp([ W.text number_into_string(W.n, W.N) '/' num2str(W.N) ' [elapsed time ' sectohhmmss(W.time_elapsed, 'short')  ' | ??:??:?? remaining]']);
    end
end
    