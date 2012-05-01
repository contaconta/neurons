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
        R.CellTimeInfo(t).TotalCableLengthTracked          = zeros(size(seq));
        R.CellTimeInfo(t).TotalCableLengthFilopodiaTracked = zeros(size(seq));
        R.CellTimeInfo(t).BranchCountTracked               = zeros(size(seq));
        R.CellTimeInfo(t).FiloCountTracked                 = zeros(size(seq));
        
        R.CellTimeInfo(t).TotalCableLengthAll              = zeros(size(seq));
        R.CellTimeInfo(t).TotalCableLengthFilopodiaAll     = zeros(size(seq));
        R.CellTimeInfo(t).BranchCountAll                   = zeros(size(seq));
        R.CellTimeInfo(t).FiloCountAll                     = zeros(size(seq));
        
        R.CellTimeInfo(t).NumTrackedNeurites               = zeros(size(seq));
        R.CellTimeInfo(t).NumNeuritesAll                   = zeros(size(seq));
        
        R.CellTimeInfo(t).TotalF_Actin                     = zeros(size(seq));
        R.CellTimeInfo(t).MeanF_Actin                      = zeros(size(seq));
        R.CellTimeInfo(t).MeanFiloLength                   = zeros(size(seq));
        
        % Cumulted over all detected neurites
        R.CellTimeInfo(t).TotalCableLength2                = zeros(size(seq));
        R.CellTimeInfo(t).TotalCableLengthNoFilopodia2     = zeros(size(seq));
        R.CellTimeInfo(t).BranchesLengthsDistrib           = cell(size(seq));
        R.CellTimeInfo(t).BranchesFilopodiaFlags           = cell(size(seq));
        
        
        for i = 1:length(seq)
            d = seq(i);
            R.CellTimeInfo(t).TotalCableLengthAll(i)          = length(find(R.FILAMENTS(d).NeuriteID > 0));
            R.CellTimeInfo(t).TotalCableLengthFilopodiaAll(i) = sum(R.FILAMENTS(d).FilopodiaFlag);
            R.CellTimeInfo(t).BranchCountAll(i)               = length(find(R.FILAMENTS(d).NumKids(R.FILAMENTS(d).NeuriteID > 0) > 1));
            R.CellTimeInfo(t).FiloCountAll(i)                 = length(R.FILAMENTS(d).FilopodiaLengths);
            R.CellTimeInfo(t).NumTrackedNeurites(i)           = length(R.FILAMENTS(d).NTrackedList);
            R.CellTimeInfo(t).NumNeuritesAll(i)               = length(R.FILAMENTS(d).NIdxList);
            
            R.CellTimeInfo(t).TotalF_Actin(i)                 = R.FILAMENTS(d).FilopodiaTotalF_Actin;
            R.CellTimeInfo(t).MeanF_Actin(i)                  = R.FILAMENTS(d).FilopodiaMeanF_Actin;
            R.CellTimeInfo(t).MeanFiloLength(i)               = R.FILAMENTS(d).FilopodiaMeanLengths;
            
            R.CellTimeInfo(t).TotalCableLength2(i)            = R.FILAMENTS(d).FethTotalCableLength;
            R.CellTimeInfo(t).TotalCableLengthNoFilopodia2(i) = R.FILAMENTS(d).FethTotalCableLengthWithoutFilo;
            R.CellTimeInfo(t).BranchesLengthsDistrib{i}       = R.FILAMENTS(d).FethBranchesLengths;
            R.CellTimeInfo(t).BranchesFilopodiaFlags{i}       = R.FILAMENTS(d).FilopodiaFlags;
        end
        
        [expContractVector, timeExpanding, timeContracting, numberInflexionPoints, freqExpansion] = trkTemporalAnalysisVector(R.CellTimeInfo(t).TotalCableLengthAll); %#ok<ASGLU>
        R.CellTimeInfo(t).TotalCableLengthAllTimeExpanding = timeExpanding;
        R.CellTimeInfo(t).TotalCableLengthAllTimeContracting = timeContracting;
        R.CellTimeInfo(t).TotalCableLengthAllFreqExpansion = freqExpansion;
        
        [expContractVector, timeExpanding, timeContracting, numberInflexionPoints, freqExpansion] = trkTemporalAnalysisVector(R.CellTimeInfo(t).NumTrackedNeurites); %#ok<ASGLU>
        R.CellTimeInfo(t).NumTrackedNeuritesTimeExpanding = timeExpanding;
        R.CellTimeInfo(t).NumTrackedNeuritesTimeContracting = timeContracting;
        R.CellTimeInfo(t).NumTrackedNeuritesFreqExpansion = freqExpansion;
        
        [expContractVector, timeExpanding, timeContracting, numberInflexionPoints, freqExpansion] = trkTemporalAnalysisVector(R.CellTimeInfo(t).NumNeuritesAll); %#ok<ASGLU>
        R.CellTimeInfo(t).NumNeuritesAllTimeExpanding = timeExpanding;
        R.CellTimeInfo(t).NumNeuritesAllTimeContracting = timeContracting;
        R.CellTimeInfo(t).NumNeuritesAllFreqExpansion = freqExpansion;
        
        [expContractVector, timeExpanding, timeContracting, numberInflexionPoints, freqExpansion] = trkTemporalAnalysisVector(R.CellTimeInfo(t).TotalCableLengthFilopodiaAll); %#ok<ASGLU>
        R.CellTimeInfo(t).TotalCableLengthFilopodiaAllTimeExpanding = timeExpanding;
        R.CellTimeInfo(t).TotalCableLengthFilopodiaAllTimeContracting = timeContracting;
        R.CellTimeInfo(t).TotalCableLengthFilopodiaAllFreqExpansion = freqExpansion;
        
        
        [expContractVector, timeExpanding, timeContracting, numberInflexionPoints, freqExpansion] = trkTemporalAnalysisVector(R.CellTimeInfo(t).BranchCountAll); %#ok<ASGLU>
        R.CellTimeInfo(t).BranchCountAllTimeExpanding = timeExpanding;
        R.CellTimeInfo(t).BranchCountAllTimeContracting = timeContracting;
        R.CellTimeInfo(t).BranchCountAllFreqExpansion = freqExpansion;
        
        [expContractVector, timeExpanding, timeContracting, numberInflexionPoints, freqExpansion] = trkTemporalAnalysisVector(R.CellTimeInfo(t).FiloCountAll); %#ok<ASGLU>
        R.CellTimeInfo(t).FiloCountAllTimeExpanding = timeExpanding;
        R.CellTimeInfo(t).FiloCountAllTimeContracting = timeContracting;
        R.CellTimeInfo(t).FiloCountAllFreqExpansion = freqExpansion;
        
        
        [expContractVector, timeExpanding, timeContracting, numberInflexionPoints, freqExpansion] = trkTemporalAnalysisVector(R.CellTimeInfo(t).TotalF_Actin); %#ok<ASGLU>
        R.CellTimeInfo(t).TotalF_ActinTimeExpanding = timeExpanding;
        R.CellTimeInfo(t).TotalF_ActinTimeContracting = timeContracting;
        R.CellTimeInfo(t).TotalF_ActinFreqExpansion = freqExpansion;
        
        [expContractVector, timeExpanding, timeContracting, numberInflexionPoints, freqExpansion] = trkTemporalAnalysisVector(R.CellTimeInfo(t).MeanF_Actin); %#ok<ASGLU>
        R.CellTimeInfo(t).MeanF_ActinTimeExpanding = timeExpanding;
        R.CellTimeInfo(t).MeanF_ActinTimeContracting = timeContracting;
        R.CellTimeInfo(t).MeanF_ActinFreqExpansion = freqExpansion;
        
        [expContractVector, timeExpanding, timeContracting, numberInflexionPoints, freqExpansion] = trkTemporalAnalysisVector(R.CellTimeInfo(t).MeanFiloLength); %#ok<ASGLU>
        R.CellTimeInfo(t).MeanFiloLengthTimeExpanding = timeExpanding;
        R.CellTimeInfo(t).MeanFiloLengthTimeContracting = timeContracting;
        R.CellTimeInfo(t).MeanFiloLengthFreqExpansion = freqExpansion;
        
        [expContractVector, timeExpanding, timeContracting, numberInflexionPoints, freqExpansion] = trkTemporalAnalysisVector(R.CellTimeInfo(t).TotalCableLength2); %#ok<ASGLU>
        R.CellTimeInfo(t).TotalCableLength2TimeExpanding = timeExpanding;
        R.CellTimeInfo(t).TotalCableLength2TimeContracting = timeContracting;
        R.CellTimeInfo(t).TotalCableLength2FreqExpansion = freqExpansion;
        
        [expContractVector, timeExpanding, timeContracting, numberInflexionPoints, freqExpansion] = trkTemporalAnalysisVector(R.CellTimeInfo(t).TotalCableLengthNoFilopodia2); %#ok<ASGLU>
        R.CellTimeInfo(t).TotalCableLengthNoFilopodia2TimeExpanding = timeExpanding;
        R.CellTimeInfo(t).TotalCableLengthNoFilopodia2TimeContracting = timeContracting;
        R.CellTimeInfo(t).TotalCableLengthNoFilopodia2FreqExpansion = freqExpansion;
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
        [expV, timeE, timeC, numberI, freqE] = trkTemporalAnalysisVector(MajorAxisLength); %#ok<*ASGLU,ASGLU>
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

            R.CellTimeInfo(dID).TotalCableLengthTracked(di) = R.CellTimeInfo(dID).TotalCableLengthTracked(di) + R.N(n).TotalCableLength;
            R.CellTimeInfo(dID).TotalCableLengthFilopodiaTracked(di) = R.CellTimeInfo(dID).TotalCableLengthFilopodiaTracked(di) + R.N(n).FiloCableLength;
            R.CellTimeInfo(dID).BranchCountTracked(di) = R.CellTimeInfo(dID).BranchCountTracked(di) + R.N(n).BranchCount;
            R.CellTimeInfo(dID).FiloCountTracked(di) = R.CellTimeInfo(dID).FiloCountTracked(di) + R.N(n).FiloCount;
            
            R.CellTimeInfo(dID).TrackedFethTotalCableLength(di)                   = R.CellTimeInfo(dID).TrackedFethTotalCableLength(di) + R.N(n).FethTotalCableLength;
            R.CellTimeInfo(dID).TrackedFethTotalCableLengthWithoutFilopodia(di)   = R.CellTimeInfo(dID).TrackedFethTotalCableLengthWithoutFilopodia(di) + R.N(n).FethTotalCableLengthWithoutFilopodia;
            R.CellTimeInfo(dID).TrackedFethBranchesLengths{di}                    = [R.CellTimeInfo(dID).TrackedFethBranchesLengths{di} R.N(n).BranchLengthsDistribution];
            R.CellTimeInfo(dID).TrackedFethBranchesFilopodiaFlags{di}             = [R.CellTimeInfo(dID).TrackedFethBranchesFilopodiaFlags{di} R.N(n).FilopodiaFlags];
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




R.D(1).TotalCableLengthTrackedExpand = [];


numTracks = length(R.trkSeq);
for t = 1:numTracks

    seq = R.trkSeq{t};
    
    if ~isempty(seq)
        
        TotalCableLengthTracked = R.CellTimeInfo(t).TotalCableLengthTracked;
        x = -3:1:3;
        sigma = 1.0;
        filt = exp(-x.*x/(2*sigma*sigma))/sqrt(2*pi*sigma*sigma);
        TotalCableLengthTracked=imfilter(TotalCableLengthTracked, filt, 'same', 'replicate');
        [expContractVector, timeExpanding, timeContracting, numberInflexionPoints, freqExpansion] = trkTemporalAnalysisVector(TotalCableLengthTracked); %#ok<*ASGLU>
        for i = 1:length(seq)
            d = seq(i);
            R.D(d).TotalCableLengthTrackedExpand = expContractVector(i);
        end
        R.CellTimeInfo(t).TotalCableLengthTrackedTimeExpanding = timeExpanding;
        R.CellTimeInfo(t).TotalCableLengthTrackedTimeContracting = timeContracting;
        R.CellTimeInfo(t).TotalCableLengthTrackedFreqExpansion = freqExpansion;
        
        TotalCableLengthFilopodiaTracked = R.CellTimeInfo(t).TotalCableLengthFilopodiaTracked;
        x = -3:1:3;
        sigma = 1.0;
        filt = exp(-x.*x/(2*sigma*sigma))/sqrt(2*pi*sigma*sigma);
        TotalCableLengthFilopodiaTracked=imfilter(TotalCableLengthFilopodiaTracked, filt, 'same', 'replicate');
        [expContractVector, timeExpanding, timeContracting, numberInflexionPoints, freqExpansion] = trkTemporalAnalysisVector(TotalCableLengthFilopodiaTracked); %#ok<*ASGLU>
        for i = 1:length(seq)
            d = seq(i);
            R.D(d).TotalCableLengthTrackedExpand = expContractVector(i);
        end
        R.CellTimeInfo(t).TotalCableLengthFilopodiaTrackedTimeExpanding = timeExpanding;
        R.CellTimeInfo(t).TotalCableLengthFilopodiaTrackedTimeContracting = timeContracting;
        R.CellTimeInfo(t).TotalCableLengthFilopodiaTrackedFreqExpansion = freqExpansion;
        
        BranchCountTracked = R.CellTimeInfo(t).BranchCountTracked;
        x = -3:1:3;
        sigma = 1.0;
        filt = exp(-x.*x/(2*sigma*sigma))/sqrt(2*pi*sigma*sigma);
        BranchCountTracked=imfilter(BranchCountTracked, filt, 'same', 'replicate');
        [expContractVector, timeExpanding, timeContracting, numberInflexionPoints, freqExpansion] = trkTemporalAnalysisVector(BranchCountTracked); %#ok<*ASGLU>
        for i = 1:length(seq)
            d = seq(i);
            R.D(d).TotalCableLengthTrackedExpand = expContractVector(i);
        end
        R.CellTimeInfo(t).BranchCountTrackedTimeExpanding = timeExpanding;
        R.CellTimeInfo(t).BranchCountTrackedTimeContracting = timeContracting;
        R.CellTimeInfo(t).BranchCountTrackedFreqExpansion = freqExpansion;
        
        
        FiloCountTracked = R.CellTimeInfo(t).FiloCountTracked;
        x = -3:1:3;
        sigma = 1.0;
        filt = exp(-x.*x/(2*sigma*sigma))/sqrt(2*pi*sigma*sigma);
        FiloCountTracked=imfilter(FiloCountTracked, filt, 'same', 'replicate');
        [expContractVector, timeExpanding, timeContracting, numberInflexionPoints, freqExpansion] = trkTemporalAnalysisVector(FiloCountTracked); %#ok<*ASGLU>
        for i = 1:length(seq)
            d = seq(i);
            R.D(d).TotalCableLengthTrackedExpand = expContractVector(i);
        end
        R.CellTimeInfo(t).FiloCountTrackedTimeExpanding = timeExpanding;
        R.CellTimeInfo(t).FiloCountTrackedTimeContracting = timeContracting;
        R.CellTimeInfo(t).FiloCountTrackedFreqExpansion = freqExpansion;
        
        
%         FethNeuriteOnlyCableLength = R.CellTimeInfo(t).FethNeuriteOnlyCableLength;
%         x = -3:1:3;
%         sigma = 1.0;
%         filt = exp(-x.*x/(2*sigma*sigma))/sqrt(2*pi*sigma*sigma);
%         FethNeuriteOnlyCableLength = imfilter(FethNeuriteOnlyCableLength, filt, 'same', 'replicate');
%         [expContractVector, timeExpanding, timeContracting, numberInflexionPoints, freqExpansion] = trkTemporalAnalysisVector(FethNeuriteOnlyCableLength); %#ok<*ASGLU>
%         for i = 1:length(seq)
%             d = seq(i);
%             R.D(d).FethNeuriteOnlyCableLengthExpand = expContractVector(i);
%         end
%         R.CellTimeInfo(t).FethNeuriteOnlyCableLengthTimeExpanding = timeExpanding;
%         R.CellTimeInfo(t).FethNeuriteOnlyCableLengthTimeContracting = timeContracting;
%         R.CellTimeInfo(t).FethNeuriteOnlyCableLengthFreqExpansion = freqExpansion;
    end
    
end