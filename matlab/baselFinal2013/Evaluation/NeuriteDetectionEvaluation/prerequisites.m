addpath(addpath(genpath([pwd '/../../'])));

if isempty( strfind(path, [pwd '/ConvertSWCFilesToOBJ/']) )
    addpath([pwd '/ConvertSWCFilesToOBJ/']);
end

if isempty( strfind(path, [pwd '/../../IO']) )
    addpath([pwd '/../../IO']);
end

if isempty( strfind(path, [pwd '/../../frangi_filter_version2a']) )
    addpath([pwd '/../../frangi_filter_version2a']);
end

if isempty( strfind(path, [pwd '/../../gaimc']) )
    addpath([pwd '/../../gaimc']);
end

if isempty( strfind(path, [pwd '/../../Geodesics']) )
    addpath([pwd '/../../Geodesics']);
end

run('../../vlfeat-0.9.16/toolbox/vl_setup');