%% make time-dependent measurements
function [D Soma] = timeMeasurements(trkSeq, timeSeq, D, Soma)

for i = 1:length(trkSeq)
    dseq = trkSeq{i};
    tseq = timeSeq{i};

    if ~isempty(dseq)

        d1 = dseq(1);
        D(d1).deltaArea = 0;
        D(d1).deltaPerimeter = 0;
        D(d1).deltaMeanGreenIntensity = 0;
        D(d1).deltaEccentricity = 0;
        D(d1).Speed = 0;
        D(d1).Acc = 0;
        D(d1).TravelDistance = 0;

        Soma(d1).deltaArea = 0;
        Soma(d1).deltaPerimeter = 0;
        Soma(d1).deltaMeanGreenIntensity = 0;
        Soma(d1).deltaEccentricity = 0;
        Soma(d1).Speed = 0;
        Soma(d1).Acc = 0;
        Soma(d1).TravelDistance = 0;

        for t = 2:length(dseq)
            d2 = dseq(t);
            d1 = dseq(t-1);
            t2 = tseq(t);
            t1 = tseq(t-1);

            D(d2).deltaArea = D(d2).Area - D(d1).Area;
            D(d2).deltaPerimeter = D(d2).Perimeter - D(d1).Perimeter;
            D(d2).deltaMeanGreenIntensity = D(d2).MeanGreenIntensity - D(d1).MeanGreenIntensity;
            D(d2).deltaEccentricity = D(d2).Eccentricity - D(d1).Eccentricity;
            D(d2).Speed = sqrt( (D(d2).Centroid(1) - D(d1).Centroid(1))^2 + (D(d2).Centroid(2) - D(d1).Centroid(2))^2) / abs(t2 -t1);
            D(d2).Acc = D(d2).Speed - D(d1).Speed;
            D(d2).TravelDistance = D(d1).TravelDistance + sqrt( (D(d2).Centroid(1) - D(d1).Centroid(1))^2 + (D(d2).Centroid(2) - D(d1).Centroid(2))^2 );


            Soma(d2).deltaArea = Soma(d2).Area - Soma(d1).Area;
            Soma(d2).deltaPerimeter = Soma(d2).Perimeter - Soma(d1).Perimeter;
            Soma(d2).deltaMeanGreenIntensity = Soma(d2).MeanGreenIntensity - Soma(d1).MeanGreenIntensity;
            Soma(d2).deltaEccentricity = Soma(d2).Eccentricity - Soma(d1).Eccentricity;
            Soma(d2).Speed = sqrt( (Soma(d2).Centroid(1) - Soma(d1).Centroid(1))^2 + (Soma(d2).Centroid(2) - Soma(d1).Centroid(2))^2) / abs(t2 -t1);
            Soma(d2).Acc = Soma(d2).Speed - Soma(d1).Speed;
            Soma(d2).TravelDistance = Soma(d1).TravelDistance + sqrt( (Soma(d2).Centroid(1) - Soma(d1).Centroid(1))^2 + (Soma(d2).Centroid(2) - Soma(d1).Centroid(2))^2 );

        end
    end
end
