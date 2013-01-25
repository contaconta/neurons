%%
trkCommonLearnSigmoids;
%%
globs = [];
% characteristics:
% 1 - somaDist, 
% 2 - somaDistNormalized, 
% 3 - nucleusDist, 
% 4 - nucleusDistNormalized, 
% 5 - centroidsDist, 
% 6 - Xing/Ng,
% 7 - lmnn2,
% 8 - importance weights
metricCounter = 0;
for curMetric = [1,5]
    metricCounter = metricCounter + 1;
    %%
    global_nmoda = [];
    global_nmodp = [];
    global_mota = [];
    global_motp = [];
    global_IDSW = [];
    for idx = 1:size(Frames, 2)
        %%
        tic
        disp('...gathering data for ksp tracking');
        % parameters
        WIND_SIZE = 3;
        TEMPORAL_WIN_SIZE    = 1;
        SPATIAL_WINDOWS_SIZE = Frames(idx).SPATIAL_WINDOWS_SIZE;
        MIN_TRACK_LENGTH     = 20;
        NB_BEST_TRACKS       = 20;
        IMAGE_SIZE           = Frames(idx).IMAGE_SIZE;
        DISTANCE_TO_BOUNDARY = 30;
        NUMBER_OF_BINS       = Frames(idx).NUMBER_OF_BINS;
        [curX, curBoth] = trkCommonCollectUnseparatedData(idx, WIND_SIZE, Frames, false);
        n = size(curX, 2);
        toc
        %% Build graph and track
        tic
        disp('...building graph');
        m = size(curBoth, 1);
        DetectedEdges = zeros(2,m);
        DetectedEdgeWeights = zeros(m,1);
        DetectedEdgeDistances = zeros(m,1);

        for i = 1:m
            DetectedEdges(1,i) = curX(curBoth(i,1),2)-1;
            DetectedEdges(2,i) = curX(curBoth(i,2),2)-1;
            DetectedEdgeWeights(i) = curBoth(i,curMetric+2); %+2 here
            DetectedEdgeDistances(i) = curBoth(i,5);
        end

        MIN_OCCUR_PROB = 1e-6;
        MAX_OCCUR_PROB = 0.999999;
        min_prob_log = log( MIN_OCCUR_PROB / (1 - MIN_OCCUR_PROB) );
        max_prob_log = log( MAX_OCCUR_PROB / (1 - MAX_OCCUR_PROB) );

        for i = 1:m
            prob_metric = ...
            1 ./ (1 + exp(-(sigmas(1,curMetric) + sigmas(2,curMetric)*DetectedEdgeWeights(i))));
            metric_weight = 0.0;
            if (prob_metric < MIN_OCCUR_PROB)
                metric_weight = min_prob_log;
            elseif (prob_metric > MAX_OCCUR_PROB)
                metric_weight = max_prob_log;
            else
                metric_weight = -log( prob_metric / (1 - prob_metric) );
            end
            DetectedEdgeWeights(i) = metric_weight;
        end
        toc

        disp('...tracking');
        tic
        [Cells, tracks, trkSeq, ~, edges] = trkCommonKShortestPaths(Frames(idx).CellsList, Frames(idx).Cells, ...
                                                                    TEMPORAL_WIN_SIZE, ...
                                                                    SPATIAL_WINDOWS_SIZE, ...
                                                                    MIN_TRACK_LENGTH, ...
                                                                    NB_BEST_TRACKS, ...
                                                                    IMAGE_SIZE, ...
                                                                    DISTANCE_TO_BOUNDARY, ...
                                                                    NUMBER_OF_BINS, DetectedEdges, DetectedEdgeWeights, DetectedEdgeDistances);
        toc
        trkSeq
        %% Get IDs
        disp('...gathering IDs from Ground Truth Cells to detected Cells');
        cellsCount = size(Frames(idx).Cells, 2);
        cellsGroundCount = size(Frames(idx).GTCells, 2);
        cellsCommonCount = size(Frames(idx).CommonCells, 2);
        fromCellsToCommon = zeros(1, cellsCount);
        fromGroundToCommon = zeros(1, cellsGroundCount);
        fromCommonToCells = zeros(1, cellsCommonCount);
        fromCommonToGround = zeros(1, cellsCommonCount);

        for i = 1:cellsCount
            % find a in a commonCellsList the cell number OR 0
            for j = Frames(idx).CommonCellsList{Frames(idx).Cells(i).Time}
                % intersection code
                if size(intersect(Frames(idx).Cells(i).SomaPixelIdxList, Frames(idx).CommonCells(j).SomaPixelIdxList), 1) == size(Frames(idx).CommonCells(j).SomaPixelIdxList,1)
                    fromCellsToCommon(i) = j;
                    fromCommonToCells(j) = i;
                    break;
                end
            end
        end

        for t = 1:Frames(idx).TMAX
            for i = Frames(idx).CellsList{t}
                for j = Frames(idx).GTCellsList{t}
                    if size(intersect(Frames(idx).Cells(i).SomaPixelIdxList, Frames(idx).GTCells(j).SomaPixelIdxList), 1)/size(Frames(idx).Cells(i).SomaPixelIdxList,1) > 0.90
                        com = fromCellsToCommon(i);
                        fromGroundToCommon(j) = com;
                        fromCommonToGround(com) = j;
                        break;
                    end
                end
            end
        end
        %% Evaluate tracking
        disp('...evaluating tracking');
        cm = 1;
        cf = 0;
        cs = 10;

        func_cm = @(x) (x*cm);
        func_cf = @(x) (x*cf);
        func_cs = @(x) (log10(x));

        trkIDs = cell(size(trkSeq));
        for i = 1:size(trkSeq,2)
            ids = [];
            for j = trkSeq{i}
                curid = 0;
                if fromCellsToCommon(j) ~= 0
                    curid = Frames(idx).CommonCells(fromCellsToCommon(j)).ID;
                end
                ids = [ids, curid];
            end
           trkIDs{i} = ids;
        end

        moda = zeros(1, Frames(idx).TMAX); % as in paper
        modp = zeros(1, Frames(idx).TMAX); % as in paper
        IDSW = zeros(1, Frames(idx).TMAX); % as in paper
        IDZV = zeros(1, Frames(idx).TMAX); % id-switches from zero (no ground truth) to nonzero
        IDVZ = zeros(1, Frames(idx).TMAX); % id-switches from nonzero to zero
        MOR  = zeros(1, Frames(idx).TMAX); % as in paper
        NM   = zeros(1, Frames(idx).TMAX); % number of matches
        NG   = zeros(1, Frames(idx).TMAX); % as in paper

        nmodaup   = zeros(1, Frames(idx).TMAX); % numerator for nmoda
        nmodadown = zeros(1, Frames(idx).TMAX); % denominator for nmoda

        motaup   = zeros(1, Frames(idx).TMAX); % numerator for mota
        motadown = zeros(1, Frames(idx).TMAX); % denominator for mota
        
        m  = zeros(1, Frames(idx).TMAX); % misses
        fp = zeros(1, Frames(idx).TMAX); % false positives, never used
        
        for i = 1:size(trkSeq, 2)
            for j = 2:size(trkSeq{i}, 2)
                v1 = trkSeq{i}(j-1);
                v2 = trkSeq{i}(j);
                t = Frames(idx).Cells(v2).Time;
                if fromCellsToCommon(v1) == 0
                    id1 = 0;
                else
                    id1 = Frames(idx).CommonCells(fromCellsToCommon(v1)).ID;
                end
                if fromCellsToCommon(v2) == 0
                    id2 = 0;
                else
                    id2 = Frames(idx).CommonCells(fromCellsToCommon(v2)).ID;
                end
                if (id1 ~= id2)
                    IDSW(t) = IDSW(t) + 1;
                    if (id1 == 0)
                        IDZV(t) = IDZV(t) + 1;
                    elseif (id2 == 0)
                        IDVZ(t) = IDVZ(t) + 1;
                    end    
                end
            end
        end

        for i = 1:size(trkSeq, 2)
            for j = 1:size(trkSeq{i}, 2)
                v1 = trkSeq{i}(j);
                t = Frames(idx).Cells(v1).Time;
                if fromCellsToCommon(v1) ~= 0
                    NM(t) = NM(t) + 1;
                    v2 = fromCommonToGround(fromCellsToCommon(v1));
                    up   = size(intersect(Frames(idx).Cells(v1).SomaPixelIdxList, Frames(idx).GTCells(v2).SomaPixelIdxList), 1);
                    down = size(union(Frames(idx).Cells(v1).SomaPixelIdxList, Frames(idx).GTCells(v2).SomaPixelIdxList), 1);
                    MOR(t) = MOR(t) + up/down;
                end
            end
        end
        
        for t = 1:Frames(idx).TMAX
            NG(t) = size(Frames(idx).GTCellsList{t}, 2);
            m(t) = NG(t) - NM(t);
            if NM(t) > 0
                modp(t) = MOR(t) / NM(t);
            end
            moda(t) = 1 - (func_cm(m(t))+func_cf(fp(t)))/NG(t);
            nmodaup(t) = nmodaup(t) + (func_cm(m(t))+func_cf(fp(t)));
            nmodadown(t) = nmodadown(t) + NG(t);
            motaup(t) = motaup(t) + func_cm(m(t)) + func_cf(fp(t)) + func_cs(IDSW(t)+1);
            motadown(t) = motadown(t) + NG(t);
        end
        
        nmoda = 1 - sum(nmodaup) / sum(nmodadown);
        nmodp = sum(modp) / Frames(idx).TMAX;
        mota = 1 - sum(motaup) / sum(motadown);
        motp = sum(MOR) / sum(NM);
        [nmoda, nmodp, mota, motp, sum(IDSW), sum(IDZV), sum(IDVZ)]
        global_nmoda = [global_nmoda; nmoda];
        global_nmodp = [global_nmodp; nmodp];
        global_mota = [global_mota; mota];
        global_motp = [global_motp; motp];
        global_IDSW = [global_IDSW; sum(IDSW), sum(IDZV), sum(IDVZ)];
    end
    %%
    glob = [global_nmoda, global_nmodp, global_mota, global_motp, global_IDSW];
    globs{metricCounter} = glob;
end
globs
%%