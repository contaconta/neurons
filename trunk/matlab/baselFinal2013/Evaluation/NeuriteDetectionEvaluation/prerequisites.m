if isempty( strfind(path, [pwd '/../../']) )
    addpath([pwd '/../../']);
end

if isempty( strfind(path, [pwd '/ConvertSWCFilesToOBJ/']) )
    addpath([pwd '/ConvertSWCFilesToOBJ/']);
end

if isempty( strfind(path, [pwd '/../../baselFinal2013/IO']) )
    addpath([pwd '/../../baselFinal2013/IO']);
end

if isempty( strfind(path, [pwd '/../../baselFinal2013/frangi_filter_version2a']) )
    addpath([pwd '/../../baselFinal2013/frangi_filter_version2a']);
end

if isempty( strfind(path, [pwd '/../../baselFinal2013/gaimc']) )
    addpath([pwd '/../../baselFinal2013/gaimc']);
end

if isempty( strfind(path, [pwd '/../../baselFinal2013/Geodesics']) )
    addpath([pwd '/../../baselFinal2013/Geodesics']);
end

if isempty( strfind(path, [pwd '/../../baselFinal2013/ksp']) )
    addpath([pwd '/../../baselFinal2013/ksp']);
end

if isempty( strfind(path, [pwd '/../../baselFinal2013/fpeak']) )
    addpath([pwd '/../../baselFinal2013/fpeak']);
end

run('../../vlfeat-0.9.16/toolbox/vl_setup');