function [tpr fpr] = folderROC(eFolder, GT)



% d = dir([gtFolder '*.png']);
% 
% 
% GT = [];
% 
% for i = 1:12  %length(d)
% 
%     fileRoot = regexp(d(i).name, '(\w*)[^\.]', 'match');
%     fileRoot = fileRoot{1};
% 
%     A = imread([gtFolder fileRoot '.png']); A = A(:,:,2) > 200;
%     A = A(:);
% 
%     GT = [GT; A]; %#ok<AGROW>
% 
% %     imagesc(A);
% %     pause(0.01);
% %     drawnow;
% 
% end


% loop through g


glist = load([eFolder 'gfile.txt']);

%tpr = zeros(size(glist)); fpr = zeros(size(glist));

for g = 1:length(glist)
    str = ['FIBSLICE*g' num2str(glist(g)) '*.jpg'];
    d2 = dir([eFolder str]);
    E = [];

   if isempty(d2)
	error(['could not find files with value g = ' num2str(glist(g))]);

   end

    for i = 1:length(d2)

        % read the image
        disp(['reading ' d2(i).name]);
        est = imread([eFolder d2(i).name]);
        est = est(:,:,2) > 200;
        est = est(:);

        E = [E; est];

    end
    
    GT1 = GT(1:length(E));	
    [tpr(g) fpr(g)] = rocstats(E, GT1, 'TPR', 'FPR');
    disp([ 'tpr: ' num2str(tpr(g))  '   fpr:  ' num2str(fpr(g)) ]);
    
%     plot(fpr(1:g), tpr(1:g));
%     drawnow;  pause(0.05);
end
    



%'/osshare/Work/neurons/matlab/features/rays/unary3roc/;
