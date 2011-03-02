function WD = trkDetectionDistance(d1,d2, WT, WSH)


%% Area
a = abs((d1.Area - d2.Area) / (d1.Area + d2.Area));
%disp(['Area Distance            = ', num2str(a)]);

%% MajorAxisLength
ma = abs((d1.MajorAxisLength - d2.MajorAxisLength) / (d1.MajorAxisLength + d2.MajorAxisLength));
%disp(['Major Axis Distance      = ', num2str(ma)]);

%% MinorAxisLength
na = abs((d1.MinorAxisLength - d2.MinorAxisLength) / (d1.MinorAxisLength + d2.MinorAxisLength));
%disp(['Minor Axis Distance      = ', num2str(na)]);

%% Eccentricity
e = abs((d1.Eccentricity - d2.Eccentricity) / (d1.Eccentricity + d2.Eccentricity));
%disp(['Eccentricity Distance    = ', num2str(e)]);

%% Perimeter
p = abs((d1.Perimeter - d2.Perimeter) / (d1.Perimeter + d2.Perimeter));
%disp(['Perimeter Distance       = ', num2str(p)]);

%% Mean Intensity
i = abs((d1.MeanIntensity - d2.MeanIntensity) / (d1.MeanIntensity + d2.MeanIntensity));
%disp(['Mean Intensity Distance  = ', num2str(i)]);

time_d  = abs(d1.Time - d2.Time);
if d1.Time == d2.Time
    time_d = Inf;
end
shape_d = a+ma+na+e+p+i;
space_d = sqrt( (d1.Centroid(1) - d2.Centroid(1))^2 + (d1.Centroid(2) - d2.Centroid(2))^2);


%disp(' ');
%disp(['[t shp spc] = [' num2str(time_d) ' ' num2str(shape_d) ' ' num2str(space_d) ']' ]);
%disp(['[wt wsh 1] = [' num2str(WT) ' ' num2str(WSH) ' 1]']);

%disp(['[wt wsh wsp] = [' num2str(WT*time_d) ' ' num2str(WSH*shape_d) ' ' num2str(space_d) ']']);


WD = WT*time_d + WSH*shape_d + space_d;
%disp(['WD = ' num2str(WD) ]);
