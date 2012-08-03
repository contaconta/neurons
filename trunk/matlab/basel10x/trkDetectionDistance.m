function WD = trkDetectionDistance(d1,d2, WT, WSH)


%% Area
a = abs((d1.Area - d2.Area) / (d1.Area + d2.Area));

%% MajorAxisLength
ma = abs((d1.MajorAxisLength - d2.MajorAxisLength) / (d1.MajorAxisLength + d2.MajorAxisLength));

%% MinorAxisLength
na = abs((d1.MinorAxisLength - d2.MinorAxisLength) / (d1.MinorAxisLength + d2.MinorAxisLength));

%% Eccentricity
e = abs((d1.Eccentricity - d2.Eccentricity) / (d1.Eccentricity + d2.Eccentricity));

%% Perimeter
p = abs((d1.Perimeter - d2.Perimeter) / (d1.Perimeter + d2.Perimeter));

%% Geometry-Based Shape distance
geoShape_d = (a + ma + na + e + p) ;

%% Mean Green Intensity
ig = abs((d1.MeanGreenIntensity - d2.MeanGreenIntensity) / (d1.MeanGreenIntensity + d2.MeanGreenIntensity));

%% Mean Red Intensity
ir = abs((d1.MeanRedIntensity - d2.MeanRedIntensity) / (d1.MeanRedIntensity + d2.MeanRedIntensity));

%% Intensity-Based distance

intShape_d = ir + ig ;

%% Time distance
time_d  = abs(d1.Time - d2.Time);
if d1.Time == d2.Time
    time_d = Inf;
end
%% shape distance
% shape_d = geoShape_d + intShape_d;

shape_d = geoShape_d + intShape_d - ig;

%% space distance
space_d = sqrt( (d1.Centroid(1) - d2.Centroid(1))^2 + (d1.Centroid(2) - d2.Centroid(2))^2);

WD = WT*time_d+ WSH*shape_d + space_d;