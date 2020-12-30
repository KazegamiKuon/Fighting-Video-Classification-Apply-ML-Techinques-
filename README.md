# Requirements
- Python 3
- PyTorch (>= 1.0)
- ffmpeg-python (https://github.com/kkroening/ffmpeg-python)

# Enviroment:
- Anaconda -Jupyter
- GPU Nvidia

# How To Use ?

Step 1: Download dataset from this: https://github.com/sayibet/fight-detection-surv-dataset

Copy data fight to fight folder and data noFight to noFight folder in folder data

Step 1.5: If you don't change and define this project for your request, you can run go to "Step 5"

Step 2: Run extract.ipynb to extract video's features.

In this step, you can chose extract after pre-process data or not. Read comment in "my_args.py" for more info.

Step 3: If you chose have time, go to folder "3Dresnext101_2s_data_feature", else go to 3Dresnext101_data_feature. Run file "combine_to_one.ipynb" to get merged data in 2 file.

Step 4: Run "RunningModel Mean.ipynb" in "model_ml" folder. It will train model Classifier. We run 11 model but you can change it to run only 1 or add other model. See that file to get more info

Step 5: Chose one of 5 version file "connect_camera" to se resualt.

0, 1, 2, 3 run only 1 model (smote knn, knn, svm, adaboost) and show only 1 window. They have same structure and run in model which was built by me so if you wanna try other model, read comment in that file

4 window is file that I run 4 model in same time. They will be shown in 4 window. I make this file to compare each model.

# Acknowledgements
The code re-used code from https://github.com/antoine77340/video_feature_extractor
