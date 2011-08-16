function bsOrganizePlateData(srcfolder, destfolder)

% dafault parameters
prefixPat = 'Experiment2';
channelsPat = {'_w2LED red_', '_w1LED green_'}; 
channelFolderNames = {'red', 'green'};
sitePat = '_s(\d*)_'; sitePrecisionStr = '%03d';
timePat = '_t(\d*)'; timePrecisionStr = '%04d';
extPat = '.TIF';




% get the source and destination folders
if ~exist('srcfolder', 'var')
    srcfolder = input('Please provide the path to the folder containing the raw images:\n', 's');
end
if ~exist('destfolder', 'var')
    destfolder = input('\nPlease provide the path to the PLATE FOLDER where the data should be organized:\n', 's');
    
    if ~exist(destfolder, 'dir')
        try
            mkdir(destfolder);
            disp(['Created PLATE FOLDER: ' destfolder]);
        catch me
            error(['Error creating PLATE FOLDER ' destfolder]);
        end
    end
end
if ~strcmp(srcfolder(end), '/')
    srcfolder = [srcfolder '/'];
end
if ~strcmp(destfolder(end), '/')
    destfolder = [destfolder '/'];
end 

% define an image file pattern used find valid filenames
tmp = input(['\nThe pattern to locate image files is:\n' prefixPat '\n\nPlease type a new pattern to replace it, or press enter to accept:\n'], 's');
if ~strcmp('', tmp)
    prefixPat = tmp;
else
    disp([prefixPat ' accepted.']);
end


% get a list of all valid files in the source directory
d = [];
for c = 1:length(channelsPat)
    dtemp = dir([ srcfolder prefixPat channelsPat{c} '*' extPat]);
    d = [d; dtemp]; %#ok<AGROW>
end

% enumerate the list of sites and channels for this plate
sites = []; channels = {};
for k = 1:numel(d)
    s_string = regexp([destfolder d(k).name], sitePat, 'match');
    num = regexp(s_string{1}, '\d*', 'match');
    num = str2num(num{1}); %#ok<ST2NM>
    sites = [sites; num]; %#ok<AGROW>
    
    for c = 1:length(channelsPat)
        a = regexpi(d(k).name, channelsPat{c}, 'match');
        if ~isempty(a)
            channels{end+1} = channelFolderNames{c}; %#ok<AGROW>
            channels = unique(channels);
        end
    end
end
sites = unique(sites);


% create the directory structure in the destination folder, and copy images
for i = 1:numel(sites)

    % create the site folder
    siteFolder = [destfolder  sprintf(sitePrecisionStr, sites(i)) '/'];

    if ~exist(siteFolder, 'dir')
        mkdir(siteFolder);
    end
    
    % create the channel subfolders
    for j = 1:length(channels)
        channelFolder = [destfolder  sprintf(sitePrecisionStr, sites(i)) '/' channels{j} '/'];
        if ~exist(channelFolder, 'dir')
            mkdir(channelFolder);
        end
    end    
    
    % get a list of files belonging to this site
    d = dir([ srcfolder prefixPat '*_s' num2str(sites(i)) '_*' extPat]);
    
    % get a list of time steps for this site
    timelist = [];
    for k = 1:length(d)
        tstr = regexp(d(k).name, timePat, 'match');
        n = regexp(tstr{1}, '\d*', 'match');
        t_k = str2num(n{end}); %#ok<ST2NM>
        timelist = unique( [timelist t_k]);
    end
    timelist = sort(timelist);
    
    % copy each time step to each channel folder
    for t = 1:numel(timelist)
        for c = 1:length(channels)
            try
                channelFolder = [destfolder  sprintf(sitePrecisionStr, sites(i)) '/' channels{c} '/'];        
                %str = ['cp "' srcfolder prefixPat channels{c} '*_s' num2str(sites(i)) '*_t' num2str(timelist(t)) extPat '" '  channelFolder sprintf(['im' timePrecisionStr], timelist(t)) '.tif'];
                
                str = ['cp "' srcfolder 'Experiment2_w1LED green_s' num2str(sites(i)) '_t' num2str(timelist(t)) extPat '" '  channelFolder sprintf(['im' timePrecisionStr], timelist(t)) '.tif'];
                
                system(str);
                
            catch me
                error(['Failed to execute copy command: ' str]);
            end            
        end
    end
end
    






keyboard;



