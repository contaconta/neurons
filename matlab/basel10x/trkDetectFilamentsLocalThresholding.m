function [FILAMENTS, Regions] = trkDetectFilamentsLocalThresholding(Somata, Tubularity)

TMAX = length(Somata);

FILAMENTS = cell(size(Somata));

% FIL = cell(size(S));
Regions = cell(size(Somata));
% L = cell(size(S));

% numberofPoints = 1000;
% % eps = 1e-7;
% delta = 3;
% sm    = 3;
% 
parfor t = 1:TMAX
    
    [U, Regions{t}, L] = RegionGrowingNeurites([1;1], Tubularity{t}, double(Somata{t}));
    [listOfCurves, listOfPossibleThesholds, listOfCandidateEndPoints, listOfForbiddenEndPoints] = getGeodesicThresholdPerRegion(U, L, Regions{t});
%     keyboard
%     [listOfCandidateEndPoints, ~] = getLocalMaximaLength(U, L, Regions{t});
    startPoints = [];
    for i = 1:length(listOfCandidateEndPoints)
        startPoints = vertcat(startPoints, listOfCandidateEndPoints{i});
    end
    if ~isempty(startPoints)
        FILAMENTS{t} = BackPropagate(startPoints', U);
    else
        FILAMENTS{t} = zeros(size(U));
    end
%     FIL = zeros(size(Somata{t}));
%     for i=1:max(Regions{t}(:))
%         Ulist = U(Regions{t} == i & ~Somata{t});
%         h = sort(Ulist);
%         du = h(round(length(h)/25));
% %         tab = logspace(log10(min(Ulist)+eps), log10(max(Ulist)), numberofPoints);
%         tab = linspace(min(Ulist), min(Ulist) + du, numberofPoints);
%         v = zeros(1, length(tab) -delta);
%         idx = 1;
%         T = tab(1:end-delta);
%         for j = 1:length(T)
%             v(idx) =  sum(Ulist> tab(j) & Ulist <= tab(j+delta) ) / sum(Ulist <=tab(j+delta));
%             idx = idx+1;
%         end
%         v = smooth(v, sm);
%         [ymax,imax,ymin,imin] = extrema( v );
%         imin = sort(imin);
%         imax = sort(imax);
%         
%         FIL( Regions{t} == i & U < T(imin(end))) = 1;
%         figure
%         plot(T,v);
%         hold on;
%         plot(T(imax),ymax,'r*',T(imin),ymin,'g*')
%         pause;
%     end
end