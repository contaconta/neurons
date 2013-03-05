function WD = trkDetectionDistance(d1,d2, WT, WSH)


%% Area
a = abs((d1.NucleusArea - d2.NucleusArea) / (d1.NucleusArea + d2.NucleusArea));
%disp(['Area Distance            = ', num2str(a)]);

%% MajorAxisLength
ma = abs((d1.NucleusMajorAxisLength - d2.NucleusMajorAxisLength) / (d1.NucleusMajorAxisLength + d2.NucleusMajorAxisLength));
%disp(['Major Axis Distance      = ', num2str(ma)]);

%% MinorAxisLength
na = abs((d1.NucleusMinorAxisLength - d2.NucleusMinorAxisLength) / (d1.NucleusMinorAxisLength + d2.NucleusMinorAxisLength));
%disp(['Minor Axis Distance      = ', num2str(na)]);

%% Eccentricity
e = abs((d1.NucleusEccentricity - d2.NucleusEccentricity) / (d1.NucleusEccentricity + d2.NucleusEccentricity));
%disp(['Eccentricity Distance    = ', num2str(e)]);

%% Perimeter
p = abs((d1.NucleusPerimeter - d2.NucleusPerimeter) / (d1.NucleusPerimeter + d2.NucleusPerimeter));
%disp(['Perimeter Distance       = ', num2str(p)]);

%% Mean Intensity
i = abs((d1.NucleusMeanGreenIntensity - d2.NucleusMeanGreenIntensity) / (d1.NucleusMeanGreenIntensity + d2.NucleusMeanGreenIntensity));
%disp(['Mean Intensity Distance  = ', num2str(i)]);

time_d  = abs(d1.Time - d2.Time);
if d1.Time == d2.Time
    time_d = Inf;
end
shape_d = a+ma+na+e+p+i;

c1 = d1.NucleusCentroid;
c2 = d2.NucleusCentroid;
space_d = distance(c1, c2);


%disp(' ');
%disp(['[t shp spc] = [' num2str(time_d) ' ' num2str(shape_d) ' ' num2str(space_d) ']' ]);
%disp(['[wt wsh 1] = [' num2str(WT) ' ' num2str(WSH) ' 1]']);

%disp(['[wt wsh wsp] = [' num2str(WT*time_d) ' ' num2str(WSH*shape_d) ' ' num2str(space_d) ']']);


WD = WT*time_d + WSH*shape_d + space_d;
%disp(['WD = ' num2str(WD) ]);
