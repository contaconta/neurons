function WD = trkNeuriteDistance(d1,d2)

weightShape = 50;
weightMass = 50;


%% MajorAxisLength
ma = abs((d1.MajorAxisLength - d2.MajorAxisLength) / (d1.MajorAxisLength + d2.MajorAxisLength));
%disp(['Major Axis Distance      = ', num2str(ma)]);

%% MinorAxisLength
na = abs((d1.MinorAxisLength - d2.MinorAxisLength) / (d1.MinorAxisLength + d2.MinorAxisLength));
%disp(['Minor Axis Distance      = ', num2str(na)]);

%% Eccentricity
e = abs((d1.Eccentricity - d2.Eccentricity) / (d1.Eccentricity + d2.Eccentricity));
%disp(['Eccentricity Distance    = ', num2str(e)]);

%% TotalMass
m = abs((d1.TotalMass - d2.TotalMass) / (d1.TotalMass + d2.TotalMass));

%% DistToSomaMedian
dm = abs((d1.DistToSomaMedian - d2.DistToSomaMedian) / (d1.DistToSomaMedian + d2.DistToSomaMedian));

%% RadialDotProd
r = abs((d1.RadialDotProd - d2.RadialDotProd) / (d1.RadialDotProd + d2.RadialDotProd));

%% DistToSomaStandDev
ds = abs((d1.DistToSomaStandDev - d2.DistToSomaStandDev) / (d1.DistToSomaStandDev + d2.DistToSomaStandDev));

%% FiloMass
fm = abs((d1.FiloMass - d2.FiloMass) / (d1.FiloMass + d2.FiloMass));

shape_d = ma+na+e+r+ds+fm;
centroid_d = sqrt( (d1.CentroidOffset(1) - d2.CentroidOffset(1))^2 + (d1.CentroidOffset(2) - d2.CentroidOffset(2))^2);
soma_d = sqrt( (d1.SomaContact(1) - d2.SomaContact(1))^2 + (d1.SomaContact(2) - d2.SomaContact(2))^2);

mass_d = m;

%disp(' ');
%disp(['[t shp spc] = [' num2str(time_d) ' ' num2str(shape_d) ' ' num2str(space_d) ']' ]);
%disp(['[wt wsh 1] = [' num2str(WT) ' ' num2str(WSH) ' 1]']);

%disp(['[wt wsh wsp] = [' num2str(WT*time_d) ' ' num2str(WSH*shape_d) ' ' num2str(space_d) ']']);


WD = weightShape*shape_d + weightMass*mass_d + centroid_d + soma_d;
%disp(['WD = ' num2str(WD) ]);
