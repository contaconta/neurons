function R = trkTimeDependentAnalysis(R)


MINIUMUM_NEURITE_LENGTH_FOR_EXPAND = 20;


%% Time-Dependent analysis for D
R.D(1).ExpandArea = [];

R.CellTimeInfo = [];
R.NTimeInfo = [];

numTracks = length(R.trkSeq);
for t = 1:numTracks

    seq = R.trkSeq{t};
    
    if ~isempty(seq)
        
        

        Area = [R.D(seq).Area];
        [expContractVector, timeExpanding, timeContracting, numberInflexionPoints, freqExpansion] = trkTemporalAnalysisVector(Area);
        
        for i = 1:length(seq)
            d = seq(i);
            R.D(d).Expand = expContractVector(i);
        end
        R.CellTimeInfo(t).NucleusTimeExpanding = timeExpanding;
        R.CellTimeInfo(t).NucleusTimeContracting = timeContracting;
        R.CellTimeInfo(t).NucleusFreqExpansion = freqExpansion;
        
        Area = [R.Soma(seq).Area];
        [expContractVector, timeExpanding, timeContracting, numberInflexionPoints, freqExpansion] = trkTemporalAnalysisVector(Area); %#ok<*ASGLU>
        for i = 1:length(seq)
            d = seq(i);
            R.Soma(d).Expand = expContractVector(i);
        end
        R.CellTimeInfo(t).SomaTimeExpanding = timeExpanding;
        R.CellTimeInfo(t).SomaTimeContracting = timeContracting;
        R.CellTimeInfo(t).SomaFreqExpansion = freqExpansion;
        
        
        % compute some stats using FILAMENTS
        R.CellTimeInfo(t).KevinTotalCableLength = zeros(size(seq));
        R.CellTimeInfo(t).KevinTotalCableLengthFilopodia = zeros(size(seq));
        R.CellTimeInfo(t).KevinBranchCount = zeros(size(seq));
        R.CellTimeInfo(t).KevinFiloCount = zeros(size(seq));
        R.CellTimeInfo(t).GermanTotalCableLength = zeros(size(seq));
        R.CellTimeInfo(t).NumTrackedNeurites = zeros(size(seq));
        R.CellTimeInfo(t).GermanNumNeurites = zeros(size(seq));
        R.CellTimeInfo(t).GermanTotalCableLengthFilopodia = zeros(size(seq));
        for i = 1:length(seq)
            d = seq(i);
            R.CellTimeInfo(t).GermanTotalCableLength(i) = length(find(R.FILAMENTS(d).NeuriteID > 0));
            R.CellTimeInfo(t).NumTrackedNeurites(i) = length(R.FILAMENTS(d).NTrackedList);
            R.CellTimeInfo(t).GermanNumNeurites(i) = length(R.FILAMENTS(d).NIdxList);
            R.CellTimeInfo(t).GermanTotalCableLengthFilopodia(i) = sum(R.FILAMENTS(d).FilopodiaFlag);
        end
        
        
        [expContractVector, timeExpanding, timeContracting, numberInflexionPoints, freqExpansion] = trkTemporalAnalysisVector(R.CellTimeInfo(t).GermanTotalCableLength); %#ok<ASGLU>
        R.CellTimeInfo(t).GermanTotalCableLengthTimeExpanding = timeExpanding;
        R.CellTimeInfo(t).GermanTotalCableLengthTimeContracting = timeContracting;
        R.CellTimeInfo(t).GermanTotalCableLengthFreqExpansion = freqExpansion;
        
        [expContractVector, timeExpanding, timeContracting, numberInflexionPoints, freqExpansion] = trkTemporalAnalysisVector(R.CellTimeInfo(t).NumTrackedNeurites); %#ok<ASGLU>
        R.CellTimeInfo(t).NumTrackedNeuritesTimeExpanding = timeExpanding;
        R.CellTimeInfo(t).NumTrackedNeuritesTimeContracting = timeContracting;
        R.CellTimeInfo(t).NumTrackedNeuritesFreqExpansion = freqExpansion;
        
        [expContractVector, timeExpanding, timeContracting, numberInflexionPoints, freqExpansion] = trkTemporalAnalysisVector(R.CellTimeInfo(t).GermanNumNeurites); %#ok<ASGLU>
        R.CellTimeInfo(t).GermanNumNeuritesTimeExpanding = timeExpanding;
        R.CellTimeInfo(t).GermanNumNeuritesTimeContracting = timeContracting;
        R.CellTimeInfo(t).GermanNumNeuritesFreqExpansion = freqExpansion;
        
        [expContractVector, timeExpanding, timeContracting, numberInflexionPoints, freqExpansion] = trkTemporalAnalysisVector(R.CellTimeInfo(t).GermanTotalCableLengthFilopodia); %#ok<ASGLU>
        R.CellTimeInfo(t).GermanTotalCableLengthFilopodiaTimeExpanding = timeExpanding;
        R.CellTimeInfo(t).GermanTotalCableLengthFilopodiaTimeContracting = timeContracting;
        R.CellTimeInfo(t).GermanTotalCableLengthFilopodiaFreqExpansion = freqExpansion;
    end
end







numTracks = length(R.trkNSeq);
for t = 1:numTracks
    seq = R.trkNSeq{t};
    if ~isempty(seq)

        % time-dependent
        BranchCount = [R.N(seq).BranchCount];
        [expV, timeE, timeC, numberI, freqE] = trkTemporalAnalysisVector(BranchCount); %#ok<ASGLU>
        R.NTimeInfo(t).BranchCountExpanding = timeE;
        R.NTimeInfo(t).BranchCountContracting = timeC;
        R.NTimeInfo(t).BranchCountFreqExpansion = freqE;

        DistToSomaExtreme = [R.N(seq).DistToSomaExtreme];
        [expV, timeE, timeC, numberI, freqE] = trkTemporalAnalysisVector(DistToSomaExtreme); %#ok<ASGLU>
        R.NTimeInfo(t).DistToSomaExtremeExpanding = timeE;
        R.NTimeInfo(t).DistToSomaExtremeContracting = timeC;
        R.NTimeInfo(t).DistToSomaExtremeFreqExpansion = freqE;
        
        MajorAxisLength = [R.N(seq).MajorAxisLength];
        [expV, timeE, timeC, numberI, freqE] = trkTemporalAnalysisVector(MajorAxisLength); %#ok<*ASGLU,ASGLU>
        R.NTimeInfo(t).MajorAxisLengthExpanding = timeE;
        R.NTimeInfo(t).MajorAxisLengthContracting = timeC;
        R.NTimeInfo(t).MajorAxisLengthFreqExpansion = freqE;
        
        x = -5:1:5;
        sigma = 1.5;
        filt = exp(-x.*x/(2*sigma*sigma))/sqrt(2*pi*sigma*sigma);
        MajorAxisLength =imfilter(MajorAxisLength, filt, 'same', 'replicate');
        [expV, timeE, timeC, numberI, freqE] = trkTemporalAnalysisVector(MajorAxisLength); %#ok<ASGLU>
        for i = 1:length(seq)
            n = seq(i);
            if R.N(n).MajorAxisLength > MINIUMUM_NEURITE_LENGTH_FOR_EXPAND
                R.N(n).Expand = expV(i);
            else
                R.N(n).Expand = 0;
            end
        end

        Eccentricity = [R.N(seq).Eccentricity];
        [expV, timeE, timeC, numberI, freqE] = trkTemporalAnalysisVector(Eccentricity); %#ok<ASGLU>
        R.NTimeInfo(t).EccentricityExpanding = timeE;
        R.NTimeInfo(t).EccentricityContracting = timeC;
        R.NTimeInfo(t).EccentricityFreqExpansion = freqE;
        
        FiloCount = [R.N(seq).FiloCount];
        [expV, timeE, timeC, numberI, freqE] = trkTemporalAnalysisVector(FiloCount); %#ok<ASGLU>
        R.NTimeInfo(t).FiloCountExpanding = timeE;
        R.NTimeInfo(t).FiloCountContracting = timeC;
        R.NTimeInfo(t).FiloCountFreqExpansion = freqE;
        
        FiloCableLength = [R.N(seq).FiloCableLength];
        [expV, timeE, timeC, numberI, freqE] = trkTemporalAnalysisVector(FiloCableLength); %#ok<ASGLU>
        R.NTimeInfo(t).FiloCableLengthExpanding = timeE;
        R.NTimeInfo(t).FiloCableLengthContracting = timeC;
        R.NTimeInfo(t).FiloCableLengthFreqExpansion = freqE;
        
        TotalCableLength = [R.N(seq).TotalCableLength];
        [expV, timeE, timeC, numberI, freqE] = trkTemporalAnalysisVector(TotalCableLength); %#ok<ASGLU>
        R.NTimeInfo(t).TotalCableLengthExpanding = timeE;
        R.NTimeInfo(t).TotalCableLengthContracting = timeC;
        R.NTimeInfo(t).TotalCableLengthFreqExpansion = freqE;

        % put some statistics into CellTimeInfo
        for i = 1:length(seq)
            n = seq(i);
            nt = R.N(n).Time;
            %d = R.N(n).NucleusD;
            dID = R.N(n).NucleusTrackID;
            di = find(R.timeSeq{dID} == nt);

            R.CellTimeInfo(dID).KevinTotalCableLength(di) = R.CellTimeInfo(dID).KevinTotalCableLength(di) + R.N(n).TotalCableLength;
            R.CellTimeInfo(dID).KevinTotalCableLengthFilopodia(di) = R.CellTimeInfo(dID).KevinTotalCableLengthFilopodia(di) + R.N(n).FiloCableLength;
            R.CellTimeInfo(dID).KevinBranchCount(di) = R.CellTimeInfo(dID).KevinBranchCount(di) + R.N(n).BranchCount;
            R.CellTimeInfo(dID).KevinFiloCount(di) = R.CellTimeInfo(dID).KevinFiloCount(di) + R.N(n).FiloCount;
        end
    end
end

R.N(1).deltaBranchCount = [];
R.N(1).deltaDistToSomaExtreme = [];
R.N(1).deltaDistToSomaStandDev = [];
R.N(1).deltaEccentricity = [];
R.N(1).deltaFiloCableLength = [];
R.N(1).deltaFiloCount = [];
R.N(1).deltaFiloPercent = [];
R.N(1).deltaMajorAxisLength = [];
R.N(1).deltaRadialDotProd = [];
R.N(1).deltaTotalCableLength = [];




numTracks = length(R.trkNSeq);
for t = 1:numTracks
    seq = R.trkNSeq{t};
    if ~isempty(seq)
        for i = 2:length(seq)
            n2 = seq(i);
            n1 = seq(i-1);
            
            R.N(n2).deltaBranchCount = R.N(n2).BranchCount - R.N(n1).BranchCount ;
            R.N(n2).deltaDistToSomaExtreme = R.N(n2).DistToSomaExtreme - R.N(n1).DistToSomaExtreme ;
            R.N(n2).deltaDistToSomaStandDev = R.N(n2).DistToSomaStandDev - R.N(n1).DistToSomaStandDev ;
            R.N(n2).deltaEccentricity = R.N(n2).Eccentricity - R.N(n1).Eccentricity ;
            R.N(n2).deltaFiloCableLength = R.N(n2).FiloCableLength - R.N(n1).FiloCableLength ;
            R.N(n2).deltaFiloCount = R.N(n2).FiloCount - R.N(n1).FiloCount ;
            R.N(n2).deltaFiloPercent = R.N(n2).FiloPercent - R.N(n1).FiloPercent ;
            R.N(n2).deltaMajorAxisLength = R.N(n2).MajorAxisLength - R.N(n1).MajorAxisLength ;
            R.N(n2).deltaRadialDotProd = R.N(n2).RadialDotProd - R.N(n1).RadialDotProd ;
            R.N(n2).deltaTotalCableLength = R.N(n2).TotalCableLength - R.N(n1).TotalCableLength ;
        end
        
        n1 = seq(1);
        n2 = seq(2);
        R.N(n1).deltaBranchCount = R.N(n2).deltaBranchCount;
        R.N(n1).deltaDistToSomaExtreme = R.N(n2).deltaDistToSomaExtreme; 
        R.N(n1).deltaDistToSomaStandDev = R.N(n2).deltaDistToSomaStandDev; 
        R.N(n1).deltaEccentricity = R.N(n2).deltaEccentricity; 
        R.N(n1).deltaFiloCableLength = R.N(n2).deltaFiloCableLength; 
        R.N(n1).deltaFiloCount = R.N(n2).deltaFiloCount;
        R.N(n1).deltaFiloPercent = R.N(n2).deltaFiloPercent; 
        R.N(n1).deltaMajorAxisLength = R.N(n2).deltaMajorAxisLength; 
        R.N(n1).deltaRadialDotProd = R.N(n2).deltaRadialDotProd; 
        R.N(n1).deltaTotalCableLength = R.N(n2).deltaTotalCableLength; 
    end
end    




R.D(1).KevinTotalCableLengthExpand = [];


numTracks = length(R.trkSeq);
for t = 1:numTracks

    seq = R.trkSeq{t};
    
    if ~isempty(seq)
        
        KevinTotalCableLength = R.CellTimeInfo(t).KevinTotalCableLength;
        x = -3:1:3;
        sigma = 1.0;
        filt = exp(-x.*x/(2*sigma*sigma))/sqrt(2*pi*sigma*sigma);
        KevinTotalCableLength=imfilter(KevinTotalCableLength, filt, 'same', 'replicate');
        [expContractVector, timeExpanding, timeContracting, numberInflexionPoints, freqExpansion] = trkTemporalAnalysisVector(KevinTotalCableLength); %#ok<*ASGLU>
        for i = 1:length(seq)
            d = seq(i);
            R.D(d).KevinTotalCableLengthExpand = expContractVector(i);
        end
        R.CellTimeInfo(t).KevinTotalCableLengthTimeExpanding = timeExpanding;
        R.CellTimeInfo(t).KevinTotalCableLengthTimeContracting = timeContracting;
        R.CellTimeInfo(t).KevinTotalCableLengthFreqExpansion = freqExpansion;
        
        KevinTotalCableLengthFilopodia = R.CellTimeInfo(t).KevinTotalCableLengthFilopodia;
        x = -3:1:3;
        sigma = 1.0;
        filt = exp(-x.*x/(2*sigma*sigma))/sqrt(2*pi*sigma*sigma);
        KevinTotalCableLengthFilopodia=imfilter(KevinTotalCableLengthFilopodia, filt, 'same', 'replicate');
        [expContractVector, timeExpanding, timeContracting, numberInflexionPoints, freqExpansion] = trkTemporalAnalysisVector(KevinTotalCableLengthFilopodia); %#ok<*ASGLU>
        for i = 1:length(seq)
            d = seq(i);
            R.D(d).KevinTotalCableLengthExpand = expContractVector(i);
        end
        R.CellTimeInfo(t).KevinTotalCableLengthFilopodiaTimeExpanding = timeExpanding;
        R.CellTimeInfo(t).KevinTotalCableLengthFilopodiaTimeContracting = timeContracting;
        R.CellTimeInfo(t).KevinTotalCableLengthFilopodiaFreqExpansion = freqExpansion;
        
        KevinBranchCount = R.CellTimeInfo(t).KevinBranchCount;
        x = -3:1:3;
        sigma = 1.0;
        filt = exp(-x.*x/(2*sigma*sigma))/sqrt(2*pi*sigma*sigma);
        KevinBranchCount=imfilter(KevinBranchCount, filt, 'same', 'replicate');
        [expContractVector, timeExpanding, timeContracting, numberInflexionPoints, freqExpansion] = trkTemporalAnalysisVector(KevinBranchCount); %#ok<*ASGLU>
        for i = 1:length(seq)
            d = seq(i);
            R.D(d).KevinTotalCableLengthExpand = expContractVector(i);
        end
        R.CellTimeInfo(t).KevinBranchCountTimeExpanding = timeExpanding;
        R.CellTimeInfo(t).KevinBranchCountTimeContracting = timeContracting;
        R.CellTimeInfo(t).KevinBranchCountFreqExpansion = freqExpansion;
        
        
        KevinFiloCount = R.CellTimeInfo(t).KevinFiloCount;
        x = -3:1:3;
        sigma = 1.0;
        filt = exp(-x.*x/(2*sigma*sigma))/sqrt(2*pi*sigma*sigma);
        KevinFiloCount=imfilter(KevinFiloCount, filt, 'same', 'replicate');
        [expContractVector, timeExpanding, timeContracting, numberInflexionPoints, freqExpansion] = trkTemporalAnalysisVector(KevinFiloCount); %#ok<*ASGLU>
        for i = 1:length(seq)
            d = seq(i);
            R.D(d).KevinTotalCableLengthExpand = expContractVector(i);
        end
        R.CellTimeInfo(t).KevinFiloCountTimeExpanding = timeExpanding;
        R.CellTimeInfo(t).KevinFiloCountTimeContracting = timeContracting;
        R.CellTimeInfo(t).KevinFiloCountFreqExpansion = freqExpansion;
        
        
        
    end
    
end