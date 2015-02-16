#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.

./svm_map_learn -c 200 train_index_file_1 ../../Java/Expert Finding/Evaluate/All/Fold1/rank/model.bin
./svm_map_classify test_index_file_1 ../../Java/Expert Finding/Evaluate/All/Fold1/ rank/model.bin ../../Java/Expert Finding/Evaluate/All/Fold1/rank/prediction

./svm_map_learn -c 200 train_index_file_2 ../../Java/Expert Finding/Evaluate/All/Fold2/rank/model.bin
./svm_map_classify test_index_file_2 ../../Java/Expert Finding/Evaluate/All/Fold2/rank/model.bin ../../Java/Expert Finding/Evaluate/All/Fold2/rank/prediction

./svm_map_learn -c 200 train_index_file_3 ../../Java/Expert Finding/Evaluate/All/Fold3/rank/model.bin
./svm_map_classify test_index_file_3 ../../Java/Expert Finding/Evaluate/All/Fold3/rank/model.bin ../../Java/Expert Finding/Evaluate/All/Fold3/rank/prediction

./svm_map_learn -c 200 train_index_file_4 ../../Java/Expert Finding/Evaluate/All/Fold4/rank/model.bin
./svm_map_classify test_index_file_4 ../../Java/Expert Finding/Evaluate/All/Fold4/rank/model.bin ../../Java/Expert Finding/Evaluate/All/Fold4/rank/prediction

../java -jar  rank.jar "All"