    predictors_train_set = shuffle_df[:predictors_train_size] #
    target_train_set = shuffle_df[:target_train_size]

    predictors_test_set = shuffle_df[predictors_train_size:]
    target_test_set = shuffle_df[target_train_size:]


y_train
253    -1
1432   -1
355    -1
520    -1
434    -1
618    -1
838    -1
586    -1
725     0

X_train
      buying  maint doors persons  lug_boot  safety
253     0.25   0.25     3       4      0.25    0.25
1432    0.25   0.25     3       2      0.25    0.25
355     0.25   0.25     3       2      0.25    0.25

X_test
      buying  maint doors persons  lug_boot  safety
904     0.25   0.25     3       4      0.25    0.25
1304    0.25   0.25     2       2      0.25    0.25

y_test
904    -1
1304   -1

X_c
dtype:dtype('float64')
max:0.11201461280434495
min:-1.0362009980258884
shape:(10,)
size:10

could not broadcast input array from shape (4,) into shape (6,)

n_classes 2
n_features 10


special variables
dtype:dtype('float64')
max:2.791971166364041
min:-3.0669876329246666
shape:(403, 10)
size:4030

self._classes
size:2
shape:(2,)
min:0
max:1
dtype:dtype('int64')
[0:2] :[0, 1]
special variables