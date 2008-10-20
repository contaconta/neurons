function newfield = namenewfield(STRUCT, field)

if isfield(STRUCT, field)
    if strfind(field, '_')
        num = str2double(field(strfind(field,'_')+1:length(field)));
        base = strtok(field, '_');
        newfield = strcat(base, '_', num2str(num + 1));
        newfield = getnewfield(STRUCT,newfield);
    else
        newfield = strcat(field, '_1');
        newfield = getnewfield(STRUCT,newfield);
    end
else
    newfield = field;
end