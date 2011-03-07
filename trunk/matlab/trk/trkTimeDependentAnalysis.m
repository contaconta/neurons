function R = trkTimeDependentAnalysis(R)


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
        [expV, timeE, timeC, numberI, freqE] = trkTemporalAnalysisVector(MajorAxisLength); %#ok<ASGLU>
        for i = 1:length(seq)
            n = seq(i);
            R.N(n).Expand = expV(i);
        end
        R.NTimeInfo(t).MajorAxisLengthExpanding = timeE;
        R.NTimeInfo(t).MajorAxisLengthContracting = timeC;
        R.NTimeInfo(t).MajorAxisLengthFreqExpansion = freqE;

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
            
%         % compute some stats using FILAMENTS
%         for i = 1:length(seq)
%             d = seq(i);
%             R.CellTimeInfo(t).GermanTotalCableLength(i) = length(find(R.FILAMENTS(d).NeuriteID > 0));
%             R.CellTimeInfo(t).NumTrackedNeurites(i) = length(R.FILAMENTS(d).NTrackedList);
%             R.CellTimeInfo(t).GermanNumNeurites(i) = length(R.FILAMENTS(d).NIdxList);
%             R.CellTimeInfo(t).GermanTotalCableLengthFilopodia(i) = sum(R.FILAMENTS(d).FilopodiaFlag);
%         end
    end
end