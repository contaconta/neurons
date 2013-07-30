function [Graph, Branches] = convertSWCDir2Obj(swcFilesDir, outputObjFile)

if isempty( strfind(path, [pwd '/ConvertSWCFilesToOBJ/']) )
    addpath([pwd '/ConvertSWCFilesToOBJ/']);
end

% First check that the directories contain the same number of files
swcListFiles = dir([swcFilesDir '*.swc']);
tic
Trees = cell(size(swcListFiles));

if( isempty(swcListFiles) )
    FID = fopen(outputObjFile,'wt');
    if( FID == -1 )
        error('Can''t open the file.');
    end
    fclose(FID);
    return;
end

for i = 1:length(swcListFiles)
    Trees{i} = load_tree([swcFilesDir swcListFiles(i).name]);
end
tt = toc;
disp(['Elapsed time for reading all the swc files is ' num2str(tt)]);
% merge trees into a single graph structure
tic

Graph = MergeGraphs(Trees);

tt = toc;
disp(['Elapsed time for merging the swc files is ' num2str(tt)]);
% write the graph to the obj format
tic
Branches = GetBranches(Graph);
tt = toc;
disp(['Elapsed time for getting the branches is ' num2str(tt)]);
%
write_obj(Graph, Branches, outputObjFile);