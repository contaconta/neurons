function register3DSequence(source)
%
%
%
%
%
%

% find the multilayer tif sequence from the SOURCE directory
if ~strcmp(source(end), '/')
    source = [source '/'];
end    
d = dir([source '*.tif']);

% loop through all but the last multilayer tif
for i = 1:length(d)-1
    
    % load stacks V1 and V2 into memory
    if i == 1        
        f1name = [source d(i).name];
        disp(['...loading ' f1name]);
        V1 = readMultiPageTiff(f1name);
        
        f2name = [source d(i+1).name];
        disp(['...loading ' f2name]);
        V2 = readMultiPageTiff(f2name);    
    else        
        V1 = V2registered;
        f1name = f2name;
        
        f2name = [source d(i+1).name];
        disp(['...loading ' f2name]);
        V2 = readMultiPageTiff(f2name);
    end
    [p1 n1 e1 v1] = fileparts(f1name); %#ok<*NASGU>
    [p2 n2 e2 v2] = fileparts(f2name);
    
    % register the stacks
    disp(['   registering ' n2 e2 ' to ' n1 e1]); 
    V2registered = register3DStacks(V1, V2);
    
    % overwrite original stacks with registered stacks
    disp(['   writing ' f2name]);
    writeMultiPageTiff(V2registered, f2name)
    
end