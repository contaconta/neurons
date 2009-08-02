clear all
learners{1} = 'HA_x1_y1_u1_v1'
learners{2} = 'HA_x1_y1_u1_v1'
[listLearners,idxLearners]=mexEnumerateLearners(learners,[24 24]);

size(listLearners)
listLearners(1:4)

idxLearners
