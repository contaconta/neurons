
folder = '/home/ksmith/data/Sinergia/Basel/14-11-2010/';

d  = dir([folder '*.TIF']);
sequences = [];
pat = '-_s(\d*)_t';
for k = 1:numel(d)
    s_string = regexp([folder d(k).name], pat, 'match');
    num = regexp(s_string{1}, '\d*', 'match');
    num = str2num(num{1});
    sequences = [sequences; num];
end
sequences = unique(sequences);
%%
for k =1:numel(sequences)
    mkdir([folder sprintf('%03d', sequences(k))])
end

for k = 1:numel(sequences)

    search_g = ['*green*-_s' num2str(k) '_t*.TIF'];
    search_r = ['*Red*-_s' num2str(k) '_t*.TIF'];

    cmd = ['mv ' folder search_g ' ' folder sprintf('%03d',sequences(k)) '/'];
    disp(cmd);
    system(cmd);
    
    cmd = ['mv ' folder search_r ' ' folder sprintf('%03d',sequences(k)) '/'];
    disp(cmd);
    system(cmd);
    
    
    folder_cleaning_14_11_2010( [folder '/' sprintf('%03d',sequences(k)) '/'] )
end



% d = dir([folder]);
% 
% for k = 3:numel(d)-1
%     
%     folder_cleaning_14_11_2010( [folder  d(k).name '/'] )
%     
% end