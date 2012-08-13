function WD = trkDetectionDistanceNucleusAndSomata(Cell1,Cell2, WT, WSH)

%% Nucleus
% Area
nucleusA = abs((Cell1.NucleusArea- Cell2.NucleusArea) / (Cell1.NucleusArea + Cell2.NucleusArea));

% perimeter
nucleusP = abs((Cell1.NucleusPerimeter - Cell2.NucleusPerimeter) / (Cell1.NucleusPerimeter + Cell2.NucleusPerimeter));

% eccentricity
nucleusE = abs((Cell1.NucleusEccentricity - Cell2.NucleusEccentricity) / (Cell1.NucleusEccentricity + Cell2.NucleusEccentricity));

% Geometry-Based Shape distance
nucleus_geoShape_d = (nucleusA + nucleusE + nucleusP) ;

% Mean Red Intensity
nucleusIntRed = abs((Cell1.NucleusMeanRedIntensity - Cell2.NucleusMeanRedIntensity) / (Cell1.NucleusMeanRedIntensity + Cell2.NucleusMeanRedIntensity));

%% Soma
% Area
somaA = abs((Cell1.SomaArea- Cell2.SomaArea) / (Cell1.SomaArea + Cell2.SomaArea));

% perimeter
somaP = abs((Cell1.SomaPerimeter - Cell2.SomaPerimeter) / (Cell1.SomaPerimeter + Cell2.SomaPerimeter));

% eccentricity
somaE = abs((Cell1.SomaEccentricity - Cell2.SomaEccentricity) / (Cell1.SomaEccentricity + Cell2.SomaEccentricity));

% Geometry-Based Shape distance
soma_geoShape_d = (somaA + somaE + somaP) ;

% Mean Red Intensity
somaIntGreen = abs((Cell1.SomaMeanGreenIntensity - Cell2.SomaMeanGreenIntensity) / (Cell1.SomaMeanGreenIntensity + Cell2.SomaMeanGreenIntensity));

%% Intensity-Based distance

intensity_d = nucleusIntRed + somaIntGreen;

%% Time distance
time_d  = abs(Cell1.Time - Cell2.Time);
if Cell1.Time == Cell2.Time
    time_d = Inf;
end
%% shape distance

shape_d = nucleus_geoShape_d + soma_geoShape_d;

%% space distance
space_d = sqrt( (Cell1.NucleusCentroid(1) - Cell2.NucleusCentroid(1))^2 + (Cell1.NucleusCentroid(2) - Cell2.NucleusCentroid(2))^2);


%% final distance
WD = WT*time_d+ WSH*(shape_d + intensity_d)+ space_d;