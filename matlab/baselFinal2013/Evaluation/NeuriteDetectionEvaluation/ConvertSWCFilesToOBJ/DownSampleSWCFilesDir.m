function [] = DownSampleSWCFilesDir(listOfInputDir, listOfDS_SWC)


if exist(listOfDS_SWC, 'dir')
    rmdir(listOfDS_SWC, 's');
end

mkdir(listOfDS_SWC);

listOfSWCFiles = dir([listOfInputDir '*.swc']);

for i =1:length(listOfSWCFiles)
    cms_ds = ['ConvertSWCFilesToOBJ/DownSampleSWC/main.exe -i "' listOfInputDir listOfSWCFiles(i).name '" -o "' listOfDS_SWC listOfSWCFiles(i).name '" -t 0.5'];
    system(cms_ds);
end



