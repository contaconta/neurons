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
        %disp(RedImageFileName);
        %disp(GreenImageFileName);
        start = findstr(RedImageFileName, '_t') + 2;
	eend  = findstr(RedImageFileName, '.TIF');
	num = str2num(RedImageFileName(start:eend));
        disp(num)
        %disp('-------------');
    catch err
        disp('what ever !!');
    end
end
%% /raid/data/store/1/C5E670E7-CF20-403C-8169-27047AFD1E9F/e1/98/d7/20121122201559351-5091/original/plate4-G6/33/red//experiment1_w2LED red_s33_t72.TIF
STR = '/raid/data/store/1/C5E670E7-CF20-403C-8169-27047AFD1E9F/e1/98/d7/20121122201559351-5091/original/plate4-G6/33/red//experiment1_w2LED red_s33_t72.TIF';
findstr(STR, '_t')
[a, b, c] = sscanf(STR, '%s %s_t%d.TIF');