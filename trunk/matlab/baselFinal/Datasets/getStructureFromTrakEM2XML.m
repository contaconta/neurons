function TrackedCells = getStructureFromTrakEM2XML(DataRootDirectory, idx, templateHeaderFile)
% simplified xml templates are supposed to be used 
% only nuclei and soma

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
    TrackedCells{i}.LifeTime = 0;
    for k =1:numel(fieldNames)
        if strcmp(fieldNames{k}, 'nucleus') || strcmp(fieldNames{k}, 'soma')
            currentField = getfield(TrackedCells{i}, fieldNames{k});%#ok
            if(TrackedCells{i}.LifeTime == 0)
                TrackedCells{i}.LifeTime = numel(currentField.listOfObjects.t2_area);
            elseif(TrackedCells{i}.LifeTime ~= numel(currentField.listOfObjects.t2_area))
                error('annoted soma and nucleus must have the same LifeTime!!');
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
                            warning('object has multiple boundary patches');%#ok
                            disp(['cell Id = ' num2str(i) ', ' fieldNames{k} ', at time ' int2str(m)])
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
                            for l =1:numel(XX)
                                X(l) = str2double(XX{l}) + Tx+1;% +1 for matlab indexins
                                Y(l) = str2double(YY{l}) + Ty+1;% +1 for matlab indexins
                            end
                            currentField.listOfObjects.t2_area{j}.XX = [currentField.listOfObjects.t2_area{j}.XX; X];
                            currentField.listOfObjects.t2_area{j}.YY = [currentField.listOfObjects.t2_area{j}.YY; Y];
                        end
                        continue;
                    end
                end
            end
            TrackedCells{i} = setfield(TrackedCells{i}, fieldNames{k}, currentField );%#ok
            clear currentField;
        end
    end
end