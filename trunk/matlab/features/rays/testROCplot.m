

% folder containing annotations
annFolder = '/osshare/DropBox/Dropbox/aurelien/mitoAnnotations/';

% folder containing SUBFOLDERS with experiments to construct ROCs
estFolders = '/osshare/Work/neurons/matlab/features/rays/unary3roc/';

% folder to write the ROC data to
destinationFolder = '/osshare/DropBox/Dropbox/aurelien/roc/example-unary3/';

% name of the ROC file to write (image + data)
filename = 'compareLogProbsUnary3';

figure; hold on;
grid on;



disp('getting the ground truth');
d = dir([annFolder '*.png']);
GT = [];
for i = 1:12  %length(d)

    fileRoot = regexp(d(i).name, '(\w*)[^\.]', 'match');
    fileRoot = fileRoot{1};

    A = imread([annFolder fileRoot '.png']); A = A(:,:,2) > 200;
    A = A(:);

    GT = [GT; A]; %#ok<AGROW>

end


d = dir(estFolders);
clear dgood;
c = 1;
for i = 1:length(d)
    if ~strcmp(d(i).name, '.') && ~strcmp(d(i).name, '..')
        if isdir([estFolders d(i).name])
            dgood(c) = d(i); %#ok<SAGROW>
            c = c+1;
        end
    end
end
d = dgood;

tpr = cell(0); fpr = cell(0);

Cols = jet(length(d));

legnames = {};

for i = 1:length(d)
    
    estFolder1 = [estFolders d(i).name '/'];
    legnames{i} = d(i).name; %#ok<SAGROW>
    
    [tpr{i} fpr{i}] = makeROC(estFolder1, GT);
    
    plot(fpr{i}, tpr{i}, '.-', 'Color', Cols(i,:), 'LineWidth', 2); hold on;
    drawnow;  pause(0.05);
end

legend(legnames, 'Location', 'SouthEast');
xlabel('False Positive Rate');
ylabel('True Positive Rate');

print(gcf, '-dpng', '-r150', [destinationFolder filename '.png']);

save([destinationFolder filename '.mat'], 'tpr', 'fpr', 'd');
