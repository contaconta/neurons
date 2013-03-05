%% generate a list of colors for rendering the results
function cols = color_list()

% cols1 = summer(6);
% cols1 = cols1(randperm(6),:);
% cols2 = summer(8);
% cols2 = cols2(randperm(8),:);
% cols3 = summer(180);
% cols3 = cols3(randperm(180),:);
% cols = [cols1; cols2; cols3];

cols1 = jet(6);
cols1 = cols1(randperm(6),:);
cols2 = jet(8);
cols2 = cols2(randperm(8),:);
cols3 = jet(250);
cols3 = cols3(randperm(250),:);
cols = [cols1; cols2; cols3];