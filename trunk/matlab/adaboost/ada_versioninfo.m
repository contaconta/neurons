n = regexp(datestr(now), '\w+', 'match');
n1 = strcat('Copyright © ', ' ', n(3), ' ', ' Kevin Smith');

INFO.appname     = 'Adaboost Image Detection Matlab Toolbox';
INFO.institution = 'CVLAB - École Polytechnique Fédérale de Lausanne';
INFO.version     = '0.83';
INFO.author      = 'Kevin Smith';
INFO.email       = 'kevin.smith@epfl.ch';
INFO.copyright   = n1{1};



disp( '###################################################################'); 
disp( ' ');
disp( '  ---- ADABOOST CASCADED CLASSIFIER TRAINING ---- ');
disp(' ');
disp(['  ' INFO.appname ]);
disp(['  version ' INFO.version ]);
disp(['  by ' INFO.author '      ' INFO.email]);
disp(['  ' INFO.institution ]);
disp (' ');
disp(['  started ' datestr(now) ]);
disp(['  DATASETS taken from ' DATASETS.filelist]);
disp (' ');
disp(['  ' INFO.copyright ]);
disp( ' ');
disp( '###################################################################'); 
disp( ' ');