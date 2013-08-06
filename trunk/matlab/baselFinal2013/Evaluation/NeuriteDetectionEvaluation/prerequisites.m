if isempty( strfind(path, [pwd '/../../']) )
    addpath([pwd '/../../']);
end

if isempty( strfind(path, [pwd '/ConvertSWCFilesToOBJ/']) )
    addpath([pwd '/ConvertSWCFilesToOBJ/']);
end

if isempty( strfind(path, [pwd '/../../../baselFinal/frangi_filter_version2a']) )
    addpath([pwd '/../../../baselFinal/frangi_filter_version2a']);
end

if isempty( strfind(path, [pwd '/../../../baselFinal/gaimc']) )
    addpath([pwd '/../../../baselFinal/gaimc']);
end

if isempty( strfind(path, [pwd '/../../../baselFinal/Geodesics']) )
    addpath([pwd '/../../../baselFinal/Geodesics']);
end

if isempty( strfind(path, [pwd '/../../../baselFinal/ksp']) )
    addpath([pwd '/../../../baselFinal/ksp']);
end

if isempty( strfind(path, [pwd '/../../../baselFinal/fpeak']) )
    addpath([pwd '/../../../baselFinal/fpeak']);
end

run('../../vlfeat-0.9.16/toolbox/vl_setup');