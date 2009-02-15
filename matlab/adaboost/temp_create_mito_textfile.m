


pos_train_path = '/osshare/Work/Data/mitochondria24/train/pos/';
neg_train_path = '/osshare/Work/Data/mitochondria24/train/neg/';
pos_test_path  = '/osshare/Work/Data/mitochondria24/test/pos/';
neg_test_path  = '/osshare/Work/Data/mitochondria24/test/neg/';
update_path = '/osshare/Work/Data/mitochondria24/slices/';
annotation_path = '/osshare/Work/Data/mitochondria24/annotations/';

file_extension = '*.png';

text_filename = 'mitochondria24.txt';


% CREATE THE POSITIVE TRAINING PATH
ada_trainingfiles(text_filename, 'new', 'train', '+', pos_train_path);


% CREATE THE NEGATIVE TRAINING PATH
ada_trainingfiles(text_filename, 'add', 'train', '-', neg_train_path);


% CREATE THE POSITIVE TEST PATH
ada_trainingfiles(text_filename, 'add', 'validation', '+', pos_test_path);

% CREATE THE NEGATIVE TEST PATH
ada_trainingfiles(text_filename, 'add', 'validation', '-', neg_test_path);

% CREATE THE UPDATE PATH
ada_trainingfiles(text_filename, 'add', 'update', '-', update_path);

% CREATE THE ANNOTATION PATH
ada_trainingfiles(text_filename, 'add', 'annotation', '+', update_path);
