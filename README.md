Image classification using traditional machine learning and deep learning models on the CIFAR-10 dataset. The dataset consists of 60000 color images with the size of 32 x 32
and has 3 channels which are R, G, and B. Each image in the dataset is labeled with 1 of the 10 mutual classes which are airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. There are a total of 6000 images for each class with 5000 images in the training dataset and 1000 images in the test dataset.

The objective is to propose a machine-learning model that can classify images with high accuracy. This study aims to compare the performance of the models from both traditional machine learning and deep learning with different enhancement methods. Various enhancements
such as hybrid, ensemble, or reinforcement learning were made to the traditional and deep learning models
to obtain the proposed model for classifying images with high accuracy.

The traditional machine learning model was evaluated based on Logistic Regression, Naive Bayes, KNN, SVM, Random Forest, LightGBM, ANN, and CNN. To find the best parameter for each model, parameter tuning has been carried out using Random Search and Bayesian Optimization. To further evaluate the model classification performance, an ensemble model also be carried out. Deep learning model also been carried out using reinforcement learning, hybrid model, and ensemble model.

Based on the experimental results, the enhanced CNN ensemble model which is the stacked CNN-ANN with LR ensemble model has the best performance. This model obtained an 83.19% accuracy score on the test set, which is the highest among all the ensemble models across the different ensemble
techniques. 
