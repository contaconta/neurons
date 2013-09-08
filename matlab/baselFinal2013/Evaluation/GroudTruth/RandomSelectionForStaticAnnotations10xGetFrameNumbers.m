FID = fopen('/home/fbenmans/SelectionStatic10X/10XStaticSelections.txt');
C = textscan(FID, '%s %d %d');
fclose(FID);

	for i = 1:length(C{1})
    		plateDirName = C{1}(i);
    plateDirName = plateDirName{1};
    inputDir = [plateDirName num2str(C{2}(i))];
    
    RedChannelDirectory     = [inputDir '/red/'];
    GreenChannelDirectory   = [inputDir '/green/'];
    Ared        = dir([RedChannelDirectory '*.TIF']);
    Agreen      = dir([GreenChannelDirectory '*.TIF']);
    try
    RedImageFileName    = [RedChannelDirectory '/' Ared(C{3}(i)).name];
    GreenImageFileName  = [GreenChannelDirectory '/' Agreen(C{3}(i)).name];
    disp(RedImageFileName);
    disp(GreenImageFileName);
    disp('-------------');
    catch err
	disp('what ever !!');
	end
end
