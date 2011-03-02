
srcfolder = '/home/ksmith/SinergiaData/Basel/14-11-2011_siRNA_screen_test_timelapse/';
destfolder = '/home/ksmith/data/Sinergia/Basel/14-11-2010/';

% determine the list of sequences
d  = dir([srcfolder '*.TIF']);
sequences = [];
pat = '-_s(\d*)_t';
for k = 1:numel(d)
    s_string = regexp([destfolder d(k).name], pat, 'match');
    num = regexp(s_string{1}, '\d*', 'match');
    num = str2num(num{1});
    sequences = [sequences; num];
end
sequences = unique(sequences);


% create folders for each sequence in the local directory
for k =1:numel(sequences)
    mkdir([destfolder sprintf('%03d', sequences(k))])
end


% copy the images to the local directory, then cleanup the folder
for k = 1:numel(sequences)

    search_g = ['*green*-_s' num2str(k) '_t*.TIF'];
    search_r = ['*Red*-_s' num2str(k) '_t*.TIF'];

    cmd = ['cp ' srcfolder search_g ' ' destfolder sprintf('%03d',sequences(k)) '/'];
    disp(cmd);
    system(cmd);
    
    cmd = ['cp ' srcfolder search_r ' ' destfolder sprintf('%03d',sequences(k)) '/'];
    disp(cmd);
    system(cmd);
    
    
    folder_cleaning_14_11_2010( [destfolder '/' sprintf('%03d',sequences(k)) '/'] )
end


