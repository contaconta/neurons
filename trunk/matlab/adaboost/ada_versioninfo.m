n = regexp(datestr(now), '\w+', 'match');
n1 = strcat('Copyright © ', ' ', n(3), ' ', ' Kevin Smith');

INFO.appname     = 'Adaboost Image Detection Matlab Toolbox';
INFO.version     = '0.6';
INFO.author      = 'Kevin Smith';
INFO.email       = 'kevin.smith@epfl.ch';
INFO.copyright   = n1{1};



disp( '###################################################################'); 
disp( ' ');
disp( '  ---- ADABOOST CASCADED CLASSIFIER TRAINING ---- ');
disp(['  ' INFO.appname ]);
disp(['  version ' INFO.version ]);
disp(['  by ' INFO.author ]);
disp(['  ' INFO.email ]);
disp (' ');
disp(['  DATASETS from ' DATASETS.filelist]);
disp(['  ' INFO.copyright ]);
disp(['  started on ' datestr(now)]);
disp( ' ');
disp( '###################################################################'); 
disp( ' ');