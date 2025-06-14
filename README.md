# Machine Learning
**Subject** : Image Classification  
**Model** : CNN, ResNet18, VGG16  
|Model|Train|Test|
|---|---|---|
|CNN 3L|[cnn3_train.py](https://github.com/Demigodd28/ML/blob/main/cnn3_train.py)|[cnn_test.py](https://github.com/Demigodd28/ML/blob/main/cnn_test.py)|
|CNN 4L|[cnn4_train.py](https://github.com/Demigodd28/ML/blob/main/cnn4_train.py)|  
|CNN 6L|[cnn6_train.py](https://github.com/Demigodd28/ML/blob/main/cnn6_train.py)|  
|CNN 8L|[cnn8_train.py](https://github.com/Demigodd28/ML/blob/main/cnn8_train.py)|  
|ResNet18|[resnet_train.py](https://github.com/Demigodd28/ML/blob/main/resnet_train.py)|[resnet18_test.py](https://github.com/Demigodd28/ML/blob/main/resnet18_test.py)|  
|VGG16|[vgg.py](https://github.com/Demigodd28/ML/blob/main/vgg.py)|

1. All the result will restore in a file named : results\_{model}\_{num_layer}l\_{num_epoch}e, such as "result_cnn3_3l10e"  
2. The result includes *Taining_summary.txt*, *Confusion matrix.png*, *Loss_curve.png*, *Training_time.png*, *Validation_accuracy.png*
3. CNN training must set parameter in the code, whereas Resnet one set the epoch when compiling.  
4. Remember to change the  relative path before compiling.  
5. Models of CNN are in [cnn_models.py](https://github.com/Demigodd28/ML/blob/main/cnn_models.py)
6. [plot.py](https://github.com/Demigodd28/ML/blob/main/plot.py) is for plotting **Training Loss** and **Validation Accuracy**
7. [plot_1.py](https://github.com/Demigodd28/ML/blob/main/plot_1.py) is for plotting **Barchart**(Accuracy vs F1) and **Heatmap**(Precision, Recall, F1)



