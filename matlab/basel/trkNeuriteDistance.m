function WD = trkNeuriteDistance(d1,d2)

weightShape = 50;
weightCableLength = 50;


%% MajorAxisLength
if d1.MajorAxisLength == 0
    d1.MajorAxisLength = 1;
end
if d2.MajorAxisLength == 0
    d2.MajorAxisLength = 1;
end
ma = abs((d1.MajorAxisLength - d2.MajorAxisLength) / (d1.MajorAxisLength + d2.MajorAxisLength));
%disp(['Major Axis Distance      = ', num2str(ma)]);

%% MinorAxisLength
if d1.MinorAxisLength == 0
    d1.MinorAxisLength = 1;
end
if d2.MinorAxisLength == 0
    d2.MinorAxisLength = 1;
end
na = abs((d1.MinorAxisLength - d2.MinorAxisLength) / (d1.MinorAxisLength + d2.MinorAxisLength));
%disp(['Minor Axis Distance      = ', num2str(na)]);

%% Eccentricity
if d1.Eccentricity == 0
    d1.Eccentricity = 1;
end
if d2.Eccentricity == 0
    d2.Eccentricity = 1;
end
e = abs((d1.Eccentricity - d2.Eccentricity) / (d1.Eccentricity + d2.Eccentricity));
%disp(['Eccentricity Distance    = ', num2str(e)]);

%% TotalCableLength
if d1.TotalCableLength == 0
    d1.TotalCableLength = 1;
end
if d2.TotalCableLength == 0
    d2.TotalCableLength = 1;
end
m = abs((d1.TotalCableLength - d2.TotalCableLength) / (d1.TotalCableLength + d2.TotalCableLength));

%% DistToSomaMedian
if d1.DistToSomaMedian == 0
    d1.DistToSomaMedian = 1;
end
if d2.DistToSomaMedian== 0
    d2.DistToSomaMedian = 1;
end
dm = abs((d1.DistToSomaMedian - d2.DistToSomaMedian) / (d1.DistToSomaMedian + d2.DistToSomaMedian));

%% RadialDotProd
if d1.RadialDotProd == 0
    d1.RadialDotProd = 1;
end
if d2.RadialDotProd== 0
    d2.RadialDotProd = 1;
end
r = abs((d1.RadialDotProd - d2.RadialDotProd) / (d1.RadialDotProd + d2.RadialDotProd));

%% DistToSomaStandDev
if d1.DistToSomaStandDev == 0
    d1.DistToSomaStandDev = 1;
end
if d2.DistToSomaStandDev== 0
    d2.DistToSomaStandDev = 1;
end
ds = abs((d1.DistToSomaStandDev - d2.DistToSomaStandDev) / (d1.DistToSomaStandDev + d2.DistToSomaStandDev));

%% FiloCableLength
if d1.FiloCableLength == 0
    d1.FiloCableLength = 1;
end
if d2.FiloCableLength == 0
    d2.FiloCableLength = 1;
end
fm = abs((d1.FiloCableLength - d2.FiloCableLength) / (d1.FiloCableLength + d2.FiloCableLength));

shape_d = ma+na+e+r+ds+fm;
centroid_d = sqrt( (d1.CentroidOffset(1) - d2.CentroidOffset(1))^2 + (d1.CentroidOffset(2) - d2.CentroidOffset(2))^2);
soma_d = sqrt( (d1.SomaContact(1) - d2.SomaContact(1))^2 + (d1.SomaContact(2) - d2.SomaContact(2))^2);

CableLength_d = m;

%disp(' ');
%disp(['[t shp spc] = [' num2str(time_d) ' ' num2str(shape_d) ' ' num2str(space_d) ']' ]);
%disp(['[wt wsh 1] = [' num2str(WT) ' ' num2str(WSH) ' 1]']);

%disp(['[wt wsh wsp] = [' num2str(WT*time_d) ' ' num2str(WSH*shape_d) ' ' num2str(space_d) ']']);


WD = weightShape*shape_d + weightCableLength*CableLength_d + centroid_d + soma_d;

if isnan(WD)
    keyboard;
end
%disp(['WD = ' num2str(WD) ]);
