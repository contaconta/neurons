function F = coverage_test(P1,P2)

% P2.x = P2.x(1:end-50);
% P2.y = P2.y(1:end-50);

areaP1 = polyarea(P1.x, P1.y);
areaP2 = polyarea(P2.x, P2.y);

[xint, yint] = polybool('intersection', P1.x, P1.y, P2.x, P2.y);
xint = xint(~isnan(xint));
yint = yint(~isnan(yint));
intersection = polyarea(xint,yint);


precision = intersection / areaP1;
recall = intersection / areaP2;

F = (2 * precision * recall) / (precision + recall);

if isnan(F)
    F = 0;
end

    


% figure; hold on;
% patch(P1.x,P1.y, 1, 'FaceColor', [.5 .5 .5], 'FaceAlpha', .5);
% patch(P2.x,P2.y, 1, 'FaceColor', [0 0 1], 'FaceAlpha', .5);
% patch(xint,yint, 1, 'FaceColor', [1 0 0], 'FaceAlpha', .5);
% 
% keyboard;