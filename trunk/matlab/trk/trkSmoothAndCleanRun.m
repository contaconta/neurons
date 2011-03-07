function R = trkSmoothAndCleanRun(R)


x = -3:1:3;
sigma = 1.0;
filt = exp(-x.*x/(2*sigma*sigma))/sqrt(2*pi*sigma*sigma);
%totalCableLengthFilt = imfilter(TotalCableLength, filt, 'same', 'replicate');

numTracks = length(R.trkSeq);
for t = 1:numTracks

    seq = R.trkSeq{t};
    
    if ~isempty(seq)
        
        
        %% smoothing of D
        sArea = imfilter( [R.D(seq).Area], filt, 'same', 'replicate');
        sMajorAxisLength = imfilter( [R.D(seq).MajorAxisLength], filt, 'same', 'replicate');
        sMinorAxisLength = imfilter( [R.D(seq).MinorAxisLength], filt, 'same', 'replicate');
        sEccentricity = imfilter( [R.D(seq).Eccentricity], filt, 'same', 'replicate');
        sPerimeter = imfilter( [R.D(seq).Perimeter], filt, 'same', 'replicate');
        sMeanGreenIntensity = imfilter( [R.D(seq).MeanGreenIntensity], filt, 'same', 'replicate');
        sMeanRedIntensity = imfilter( [R.D(seq).MeanRedIntensity], filt, 'same', 'replicate');
        
        R.D(seq(1)).deltaArea = R.D(seq(2)).deltaArea;
        sdeltaArea = imfilter( [R.D(seq).deltaArea], filt, 'same', 'replicate');
        R.D(seq(1)).deltaPerimeter = R.D(seq(2)).deltaPerimeter;
        sdeltaPerimeter = imfilter( [R.D(seq).deltaPerimeter], filt, 'same', 'replicate');
        R.D(seq(1)).deltaMeanGreenIntensity = R.D(seq(2)).deltaMeanGreenIntensity;
        sdeltaMeanGreenIntensity = imfilter( [R.D(seq).deltaMeanGreenIntensity], filt, 'same', 'replicate');
        R.D(seq(1)).deltaEccentricity = R.D(seq(2)).deltaEccentricity;
        sdeltaEccentricity = imfilter( [R.D(seq).deltaEccentricity], filt, 'same', 'replicate');
        sSpeed = imfilter( [R.D(seq).Speed], filt, 'same', 'replicate');
        sAcc = imfilter( [R.D(seq).Acc], filt, 'same', 'replicate');
         
        
        for i = 1:length(seq)
            d = seq(i);
            R.D(d).Area = sArea(i);  
            R.D(d).MajorAxisLength  = sMajorAxisLength(i);
            R.D(d).MinorAxisLength  = sMinorAxisLength(i);
            R.D(d).Eccentricity  = sEccentricity(i);
            R.D(d).Perimeter  = sPerimeter(i);
            R.D(d).MeanGreenIntensity  = sMeanGreenIntensity(i);
            R.D(d).MeanRedIntensity  = sMeanRedIntensity(i);
            R.D(d).deltaArea  = sdeltaArea(i);
            R.D(d).deltaPerimeter  = sdeltaPerimeter(i);
            R.D(d).deltaMeanGreenIntensity  = sdeltaMeanGreenIntensity(i);
            R.D(d).deltaEccentricity  = sdeltaEccentricity(i);
            R.D(d).Speed  = sSpeed(i);
            R.D(d).Acc  = sAcc(i);
        end
           
        
        
        %% smoothing of Soma
        sArea = imfilter( [R.D(seq).Area], filt, 'same', 'replicate');
        sMajorAxisLength = imfilter( [R.D(seq).MajorAxisLength], filt, 'same', 'replicate');
        sMinorAxisLength = imfilter( [R.D(seq).MinorAxisLength], filt, 'same', 'replicate');
        sEccentricity = imfilter( [R.D(seq).Eccentricity], filt, 'same', 'replicate');
        sPerimeter = imfilter( [R.D(seq).Perimeter], filt, 'same', 'replicate');
        sMeanGreenIntensity = imfilter( [R.D(seq).MeanGreenIntensity], filt, 'same', 'replicate');
       
        R.D(seq(1)).deltaArea = R.D(seq(2)).deltaArea;
        sdeltaArea = imfilter( [R.D(seq).deltaArea], filt, 'same', 'replicate');
        R.D(seq(1)).deltaPerimeter = R.D(seq(2)).deltaPerimeter;
        sdeltaPerimeter = imfilter( [R.D(seq).deltaPerimeter], filt, 'same', 'replicate');
        R.D(seq(1)).deltaMeanGreenIntensity = R.D(seq(2)).deltaMeanGreenIntensity;
        sdeltaMeanGreenIntensity = imfilter( [R.D(seq).deltaMeanGreenIntensity], filt, 'same', 'replicate');
        R.D(seq(1)).deltaEccentricity = R.D(seq(2)).deltaEccentricity;
        sdeltaEccentricity = imfilter( [R.D(seq).deltaEccentricity], filt, 'same', 'replicate');
        sSpeed = imfilter( [R.D(seq).Speed], filt, 'same', 'replicate');
        sAcc = imfilter( [R.D(seq).Acc], filt, 'same', 'replicate');
         
        
        for i = 1:length(seq)
            d = seq(i);
            R.D(d).Area = sArea(i);  
            R.D(d).MajorAxisLength  = sMajorAxisLength(i);
            R.D(d).MinorAxisLength  = sMinorAxisLength(i);
            R.D(d).Eccentricity  = sEccentricity(i);
            R.D(d).Perimeter  = sPerimeter(i);
            R.D(d).MeanGreenIntensity  = sMeanGreenIntensity(i);
            R.D(d).deltaArea  = sdeltaArea(i);
            R.D(d).deltaPerimeter  = sdeltaPerimeter(i);
            R.D(d).deltaMeanGreenIntensity  = sdeltaMeanGreenIntensity(i);
            R.D(d).deltaEccentricity  = sdeltaEccentricity(i);
            R.D(d).Speed  = sSpeed(i);
            R.D(d).Acc  = sAcc(i);
        end
        
    end
end
      

numTracks = length(R.trkNSeq);
for t = 1:numTracks

    seq = R.trkNSeq{t};
    
    if ~isempty(seq)
        %% smoothing of N
        sBranchCount = imfilter( [R.N(seq).BranchCount], filt, 'same', 'replicate');
        sDistToSomaExtreme = imfilter( [R.N(seq).DistToSomaExtreme], filt, 'same', 'replicate');
        sDistToSomaMean = imfilter( [R.N(seq).DistToSomaMean], filt, 'same', 'replicate');
        sDistToSomaMedian = imfilter( [R.N(seq).DistToSomaMedian], filt, 'same', 'replicate');
        sDistToSomaStandDev = imfilter( [R.N(seq).DistToSomaStandDev], filt, 'same', 'replicate');
        sEccentricity = imfilter( [R.N(seq).Eccentricity], filt, 'same', 'replicate');
        sFiloCount = imfilter( [R.N(seq).FiloCount], filt, 'same', 'replicate');
        sFiloCableLength = imfilter( [R.N(seq).FiloCableLength], filt, 'same', 'replicate');
        sFiloPercent = imfilter( [R.N(seq).FiloPercent], filt, 'same', 'replicate');
        sMajorAxisLength = imfilter( [R.N(seq).MajorAxisLength], filt, 'same', 'replicate');
        sMinorAxisLength = imfilter( [R.N(seq).MinorAxisLength], filt, 'same', 'replicate');
        sRadialDotProd = imfilter( [R.N(seq).RadialDotProd], filt, 'same', 'replicate');
        sTotalCableLength = imfilter( [R.N(seq).TotalCableLength], filt, 'same', 'replicate');
       
        for i = 1:length(seq)
            n = seq(i);
            R.N(n).BranchCount = sBranchCount(i);
            R.N(n).DistToSomaExtreme = sDistToSomaExtreme(i);
            R.N(n).DistToSomaMean = sDistToSomaMean(i);
            R.N(n).DistToSomaMedian = sDistToSomaMedian(i);
            R.N(n).DistToSomaStandDev = sDistToSomaStandDev(i);
            R.N(n).Eccentricity = sEccentricity(i);
            R.N(n).FiloCount = sFiloCount(i);
            R.N(n).FiloCableLength = sFiloCableLength(i);
            R.N(n).FiloPercent = sFiloPercent(i);
            R.N(n).MajorAxisLength = sMajorAxisLength(i);
            R.N(n).MinorAxisLength = sMinorAxisLength(i);
            R.N(n).RadialDotProd = sRadialDotProd(i);
            R.N(n).TotalCableLength = sTotalCableLength(i);
        end
        
    end    
end
