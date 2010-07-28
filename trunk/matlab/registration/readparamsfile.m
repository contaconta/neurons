function [A Trans] = readparamsfile(filename)


fid = fopen(filename);
for i = 1:43
    tline = fgetl(fid); %#ok<NASGU>
end
i = 1; params = zeros(1,17);
while 1
    tline = fgetl(fid);
    if ~ischar(tline), break, end
    params(i,:) = str2num(tline); %#ok<ST2NM>
    i = i + 1;
end
params = params(:,2:9);

params = convert_to_absolutePositions(params);

Trans = params(:, 1:2);
A2 = params(:, 3:end);
A(:,1,1) = A2(:,1);
A(:,1,2) = A2(:,2);
A(:,2,1) = A2(:,3);
A(:,2,2) = A2(:,4);

fclose (fid);