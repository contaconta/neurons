function h = dummy_classify_set(weak_classifier, TRAIN)


h = randn([1 length(TRAIN.class)]);

h(h >= 0) = 1;
h(h < 0) = -1;