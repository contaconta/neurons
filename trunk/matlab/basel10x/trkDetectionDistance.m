function WD = trkDetectionDistance(d1,d2, WT, WSH)


%% Area
a = abs((d1.NucleusArea- d2.NucleusArea) / (d1.NucleusArea + d2.NucleusArea));

%% MajorAxisLength
ma = abs((d1.NucleusMajorAxisLength - d2.NucleusMajorAxisLength) / (d1.NucleusMajorAxisLength + d2.NucleusMajorAxisLength));

%% MinorAxisLength
na = abs((d1.NucleusMinorAxisLength - d2.NucleusMinorAxisLength) / (d1.NucleusMinorAxisLength + d2.NucleusMinorAxisLength));

%% Eccentricity
e = abs((d1.NucleusEccentricity - d2.NucleusEccentricity) / (d1.NucleusEccentricity + d2.NucleusEccentricity));

%% Perimeter
p = abs((d1.NucleusPerimeter - d2.NucleusPerimeter) / (d1.NucleusPerimeter + d2.NucleusPerimeter));

%% Geometry-Based Shape distance
geoShape_d = (a + ma + na + e + p) ;

%% Mean Green Intensity
ig = abs((d1.NucleusMeanGreenIntensity - d2.NucleusMeanGreenIntensity) / (d1.NucleusMeanGreenIntensity + d2.NucleusMeanGreenIntensity));

%% Mean Red Intensity
ir = abs((d1.NucleusMeanRedIntensity - d2.NucleusMeanRedIntensity) / (d1.NucleusMeanRedIntensity + d2.NucleusMeanRedIntensity));

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
space_d = sqrt( (d1.NucleusCentroid(1) - d2.NucleusCentroid(1))^2 + (d1.NucleusCentroid(2) - d2.NucleusCentroid(2))^2);

WD = WT*time_d+ WSH*shape_d + space_d;