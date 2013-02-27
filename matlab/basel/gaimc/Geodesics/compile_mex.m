listOfcppFiles = dir('*.cpp');
disp('.. compiling cpp files')
for i = 1:length(listOfcppFiles)
   mex(listOfcppFiles(i).name);
end