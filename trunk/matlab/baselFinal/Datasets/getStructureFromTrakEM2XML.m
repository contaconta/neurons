function TrackedCells = getStructureFromTrakEM2XML(DataRootDirectory, idx, templateHeaderFile)
% simplified xml templates are supposed to be used
% only nuclei and soma
ImageSize = [520, 696];
strIDx = sprintf('%03d', idx);
xmlFileNameNoHeader =[DataRootDirectory strIDx '/' strIDx 'NoHeader.xml'];
xmlFileName  = [DataRootDirectory strIDx '/' strIDx '.xml'];
if(exist(xmlFileNameNoHeader, 'file'))
    system(['rm ' xmlFileNameNoHeader]);
end
cmd_cleanXMLFromHeader = ['grep -F -x -v -f ' templateHeaderFile ' ' xmlFileName ' >> ' xmlFileNameNoHeader];
system(cmd_cleanXMLFromHeader);
%%
ST  = xml2struct(xmlFileNameNoHeader);
%% First get the objects
% Get the number of tracked cells
TrackedCells = cell(size(ST.trakem2.project.cell));
TrackedCells   = cell(size(TrackedCells));
for i = 1:numel(TrackedCells)
    %     TrackedCells{i}.oid = str2double(ST.trakem2.project.cell{i}.Attributes.id);
    fieldNames = fieldnames(ST.trakem2.project.cell{i});
    for j = 1:numel(fieldNames)
        if strcmp(fieldNames{j}, 'nucleus')
            nucleus = ST.trakem2.project.cell{i}.nucleus;
            TrackedCells{i}.nucleus = nucleus;
        end
        if strcmp(fieldNames{j}, 'soma')
            Soma = ST.trakem2.project.cell{i}.soma;
            TrackedCells{i}.soma = Soma;
        end
    end
end
%% Get Frame indexes
Nb_frames = numel(ST.trakem2.t2_layer_set.t2_layer);
LayerId = cell([1 Nb_frames]);
for i = 1:Nb_frames
    LayerId{i}   = ST.trakem2.t2_layer_set.t2_layer{i}.Attributes.oid;
end
%%  ST.trakem2.project.cell{1}.nucleus.area_list.Attributes
area_lists = ST.trakem2.t2_layer_set.t2_area_list;
for i =1:numel(area_lists)
    currentAreaList = area_lists{i};
    for j = 1:numel(TrackedCells)
        fieldNames = fieldnames(TrackedCells{j});
        for k =1:numel(fieldNames)
            if strcmp(fieldNames{k}, 'nucleus')
                if(strcmp(currentAreaList.Attributes.oid, TrackedCells{j}.nucleus.area_list.Attributes.oid))
                    TrackedCells{j}.nucleus.listOfObjects = currentAreaList;
                end
            end
            if strcmp(fieldNames{k}, 'soma')
                if(strcmp(currentAreaList.Attributes.oid, TrackedCells{j}.soma.area_list.Attributes.oid))
                    TrackedCells{j}.soma.listOfObjects = currentAreaList;
                end
            end
        end
    end
end
%%
for i =1:numel(TrackedCells)
    fieldNames = fieldnames(TrackedCells{i});
    if numel(fieldNames) == 0
        disp('cell without any component')
    elseif numel(fieldNames) == 1
        currentField = getfield(TrackedCells{i}, fieldNames{1});%#ok
        disp(['Only one object added : ' fieldNames{1} '. It''s ID is ' num2str(currentField.area_list.Attributes.oid) ' .The other object is missing']);
    end
    TrackedCells{i}.LifeTime = 0;
    for k =1:numel(fieldNames)
        if strcmp(fieldNames{k}, 'nucleus') || strcmp(fieldNames{k}, 'soma')
            currentField = getfield(TrackedCells{i}, fieldNames{k});%#ok
            is_t2_area = true;
            if(~isfield(currentField.listOfObjects, 't2_area'))
                is_t2_area = false;
                disp([fieldNames{k} ' object created, but no area_list annotated: ' 'cell Id = ' num2str(i) ', ' fieldNames{k}]);
            elseif numel(currentField.listOfObjects.t2_area) == 1
                is_t2_area = false;
                disp([fieldNames{k} ' object created, only one area_list annotated: ' 'cell Id = ' num2str(i) ', ' fieldNames{k}]);
            end
            
            if(is_t2_area)
                if(TrackedCells{i}.LifeTime == 0)
                    TrackedCells{i}.LifeTime = numel(currentField.listOfObjects.t2_area);
                elseif(TrackedCells{i}.LifeTime ~= numel(currentField.listOfObjects.t2_area))
                    %                     disp(['annotated soma and nucleus must have the same LifeTime!! ' ...
                    %                         ' life time of the Nucleus is ' num2str(TrackedCells{i}.LifeTime) ...
                    %                         ', life time of the Soma is ' num2str(numel(currentField.listOfObjects.t2_area)) ...
                    %                         ', current soma Id is ' num2str(currentField.area_list.Attributes.oid)]);
                end
                Trasform = currentField.listOfObjects.Attributes.transform;
                Trasform = Trasform(8:end-1);
                Trasform = strread(Trasform, '%s', 'delimiter', ',');
                Tx = str2double(Trasform{end-1});
                Ty = str2double(Trasform{end});
                for j=1:numel(currentField.listOfObjects.t2_area)
                    for m =1:Nb_frames
                        if(strcmp(currentField.listOfObjects.t2_area{j}.Attributes.layer_id, LayerId{m}))
                            currentField.listOfObjects.t2_area{j}.Time = m;
                            currentField.listOfObjects.t2_area{j}.XX   = [];
                            currentField.listOfObjects.t2_area{j}.YY   = [];
                            if length(currentField.listOfObjects.t2_area{j}.t2_path) > 1
                                disp(['object has multiple boundary patches.' ' Object Id = ' num2str(currentField.area_list.Attributes.oid) ', ' fieldNames{k} ', at time ' int2str(m)])
                            end
                            for kk =1:length(currentField.listOfObjects.t2_area{j}.t2_path)
                                if(length(currentField.listOfObjects.t2_area{j}.t2_path) > 1)
                                    Loc = currentField.listOfObjects.t2_area{j}.t2_path{kk}.Attributes.d;
                                else
                                    Loc = currentField.listOfObjects.t2_area{j}.t2_path.Attributes.d;
                                end
                                Loc = Loc(3:end-2); Loc = ['L ' Loc];%#ok
                                [XX, YY] = strread(Loc, 'L %s %s', 'delimiter', ' ');
                                X = zeros(size(XX)); Y = zeros(size(YY));
                                areAllPatchPointsInside = true;
                                for l =1:numel(XX)
                                    X(l) = str2double(XX{l}) + Tx+1;% +1 for matlab indexins
                                    Y(l) = str2double(YY{l}) + Ty+1;% +1 for matlab indexins
                                    if(X(l) > ImageSize(1) || X(l) < 1 || Y(l) > ImageSize(2) || Y(l) < 1)
                                        areAllPatchPointsInside = false;
                                    end
                                end
                                % to cycle
                                X(end+1) = X(1); Y(end+1) = Y(1);%#ok
                                if(~areAllPatchPointsInside && 0)
                                    disp(['Object outside image boundary. Ob ID is ' num2str(currentField.area_list.Attributes.oid) ', at time frame ' num2str(m)]);
                                end
                                % TODO: get the approprite region
                                minX = min(X);
                                maxX = max(X);
                                minY = min(Y);
                                maxY = max(Y);
                                Y = Y - minY + 1;
                                X = X - minX + 1;
                                
                                [YY, XX] = meshgrid(minY:maxY, minX:maxX);
                                XX = XX'; YY = YY';
                                localIm = false(size(XX));
                                for l = 1:numel(X)-1
                                    localIm(min(Y(l), Y(l+1)):max(Y(l), Y(l+1)), min(X(l), X(l+1)):max(X(l), X(l+1))) = true;
                                end
                                localIm = bwfill(localIm, 'holes');
                                PixelsX = XX(localIm);
                                PixelsY = YY(localIm);
                                PixelsXX = PixelsX(PixelsX > 0 & PixelsY > 0 & PixelsX <= ImageSize(2) & PixelsY <= ImageSize(1));
                                PixelsYY = PixelsY(PixelsX > 0 & PixelsY > 0 & PixelsX <= ImageSize(2) & PixelsY <= ImageSize(1));
                                Y = Y + minY - 1;
                                X = X + minX - 1;
                                
                                currentField.listOfObjects.t2_area{j}.XX = [currentField.listOfObjects.t2_area{j}.XX; X];
                                currentField.listOfObjects.t2_area{j}.YY = [currentField.listOfObjects.t2_area{j}.YY; Y];
                                currentField.listOfObjects.t2_area{j}.PixelIdxList = sub2ind(ImageSize, PixelsYY, PixelsXX);
                            end
                            continue;
                        end
                    end
                end
                TrackedCells{i} = setfield(TrackedCells{i}, fieldNames{k}, currentField );%#ok
            end
            clear currentField;
        end
    end
    
    if numel(fieldNames) == 2
        nucleusTimeFrames    = zeros(size(TrackedCells{i}.nucleus.listOfObjects.t2_area));
        somaTimeFrames      = zeros(size(TrackedCells{i}.soma.listOfObjects.t2_area));
        for k =1:numel(fieldNames)
            currentField = getfield(TrackedCells{i}, fieldNames{k});%#ok
            is_t2_area = true;
            if(~isfield(currentField.listOfObjects, 't2_area'))
                is_t2_area = false;
                disp([fieldNames{k} ' object created, but no area_list annotated: ' 'cell Id = ' num2str(i) ', ' fieldNames{k}]);
            elseif numel(currentField.listOfObjects.t2_area) == 1
                is_t2_area = false;
                disp([fieldNames{k} ' object created, only one area_list annotated: ' 'cell Id = ' num2str(i) ', ' fieldNames{k}]);
            end
            
            if(is_t2_area)
                for j=1:numel(currentField.listOfObjects.t2_area)
                    if strcmp(fieldNames{k}, 'nucleus')
                        nucleusTimeFrames(j) = currentField.listOfObjects.t2_area{j}.Time;
                    else
                        somaTimeFrames(j)   = currentField.listOfObjects.t2_area{j}.Time;
                    end
                end
                TrackedCells{i} = setfield(TrackedCells{i}, fieldNames{k}, currentField );%#ok
            end
            clear currentField;
        end
        if(numel(somaTimeFrames) ~= numel(nucleusTimeFrames))
            disp(['annotated soma and nucleus must have the same LifeTime!! ' ...
                ' life time of the Nucleus is ' num2str(numel(TrackedCells{i}.nucleus.listOfObjects.t2_area)) ...
                ', life time of the Soma is ' num2str(numel(TrackedCells{i}.soma.listOfObjects.t2_area))]);
            missingSomata  = setdiff(nucleusTimeFrames, somaTimeFrames);
            missingNuclei  = setdiff(somaTimeFrames, nucleusTimeFrames);
            if(~isempty(missingNuclei))
                disp(['Nucleus Id ' num2str(TrackedCells{i}.nucleus.area_list.Attributes.oid) ' has not been annotated at the following frames: ' num2str(sort(missingNuclei(:))')]);
            end
            if(~isempty(missingSomata))
                disp(['Soma Id ' num2str(TrackedCells{i}.soma.area_list.Attributes.oid) ' has not been annotated at the following frames: ' num2str(sort(missingSomata(:))')]);
            end
        end
        [InterSomataNuclei, Isoma, Inucleus] = intersect(somaTimeFrames, nucleusTimeFrames);
        for K = 1:numel(Isoma)
            somaIdx      = Isoma(K);
            nucleusIdx   = Inucleus(K);
            SomaPixelList    = TrackedCells{i}.soma.listOfObjects.t2_area{somaIdx}.PixelIdxList;
            NucleusPixelList = TrackedCells{i}.nucleus.listOfObjects.t2_area{nucleusIdx}.PixelIdxList;
            PixelNucleusNotInSoma = setdiff(NucleusPixelList, SomaPixelList);
            if((numel(PixelNucleusNotInSoma) / numel(NucleusPixelList)) > 0.1)
                disp(['Nucleus Id ' num2str(TrackedCells{i}.nucleus.area_list.Attributes.oid) ', at time frame ' num2str(InterSomataNuclei(K)) ', more than 10% is out of its soma']);
            end
%             if(~isempty(PixelNucleusNotInSoma))
%                 disp(['Nucleus Id ' num2str(TrackedCells{i}.nucleus.area_list.Attributes.oid) ', at time frame ' num2str(InterSomataNuclei(K)) ' is not totally included into its soma']);
%             end
            
        end
    end
    
end