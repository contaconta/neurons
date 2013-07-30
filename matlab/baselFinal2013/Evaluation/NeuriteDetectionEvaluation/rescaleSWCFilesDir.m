function [] = rescaleSWCFilesDir(h, OriginalDir, RecaledDir)

% First check that the directories contain the same number of files

if exist(RecaledDir, 'dir')
    rmdir(RecaledDir, 's');
end
mkdir(RecaledDir);


swcListFiles = dir([OriginalDir '*.swc']);

for i = 1:length(swcListFiles)
    Tree   = load_tree([OriginalDir swcListFiles(i).name]);
    Tree.X = Tree.X ./ h;
    Tree.Y = Tree.Y ./ h;
    Tree.Z = Tree.Z ./ h;
    Tree.D = Tree.D ./ h;
    swc_tree(Tree, [RecaledDir swcListFiles(i).name]);
end


