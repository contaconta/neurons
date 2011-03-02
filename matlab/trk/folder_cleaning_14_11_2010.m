function folder_cleaning_14_11_2010(folder)

work_folder = pwd;


%% find the maximum time stamp
d = dir([folder '*_t*.TIF']);
pat = '\d*';
maxt = 1;
for i = 1:length(d)
    n = regexp(d(i).name, pat, 'match');
    t_i = str2num(n{end});
    if t_i > maxt
        maxt = t_i;
    end
end


mkdir([folder 'green/']);
mkdir([folder 'red/']);



%% IMPORTANT!!!  JUST COPY THE TIFS WHERE THEY BELONG!  DO NOT LOSE ANY
%% BIT DEPTH INFORMATION!!!!

count = 1;

%% loop through the time stamps
for t=1:maxt
    
    % green channel name
    fname1 = dir( [folder '*green*_t' num2str(t) '.TIF'] );
    if ~isempty(fname1)
        fname1 = fname1.name;
    end
    
    % red channel name
    fname2 = dir( [folder '*Red*_t' num2str(t) '.TIF'] );
    if ~isempty(fname2)
        fname2 = fname2.name;
    end
    
    
    if ~isempty(fname1) && ~isempty(fname2)
        
        %str1 = ['cp ' folder fname1 ' ' folder 'green/' sprintf('im%02d', count) '.tif'];
        %disp(str1);
        %system(str1);
        copyfile([folder fname1], [folder 'green/' sprintf('im%02d', count) '.tif']);
        
        
%         str2 = ['cp ' folder fname2 ' ' folder 'red/' sprintf('im%02d', count) '.tif'];
%         disp(str2);
%         system(str2);
        copyfile([folder fname2], [folder 'red/' sprintf('im%02d', count) '.tif']);
        

        
    	count = count + 1;
    end

    
end


% cd(folder)
% 
% % zip up the original images, just in case
% str3 = ['zip original_images.zip *.TIF'];
% disp(str3);
% system(str3);
% 
% str4 = ['rm *.TIF'];
% disp(str4);
% system(str4);
% 
% cd(work_folder);

str4 = ['rm ' folder '*.TIF'];
disp(str4);
system(str4);
