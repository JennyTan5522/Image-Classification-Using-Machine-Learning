Dataset:

- The dataset chosen is the CIFAR-10 dataset. 
- The dataset consists of **60000** color images with the size of 32 x 32 and has 3 channels which are R, G and B. Each image in the dataset is labeled with 1 of the 10 classes which are airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. There are a total of 6000 images for each class with 5000 images in the train dataset and 1000 images in the test dataset

Data Preprocessing:

1) Scaling: Normalize the pixel values to a standard range within (0, 1)
   
3) Image augmentation:  Generate additional training data by applying random transformations (Horizontally flipping the image, Rotating the image, and Shifting the image left or right and up and down) to existing images, helping to improve the generalization of the model.
   
4) Histogram of oriented gradients (HOG): Capture edge and shape information in images by computing the distribution of gradient orientations.
   
5) Principal component analysis (PCA): Reduce the dimensionality of large datasets by transforming the large set of features into a smaller one while still retaining most of the information.



Framework:

![image](https://github.com/user-attachments/assets/10d13095-85f1-4711-b08d-3bd626f29598)

![image](https://github.com/user-attachments/assets/570c756a-0a45-41f8-91b4-c9ff50e5c4d1)

![image](https://github.com/user-attachments/assets/38b0f84b-1c8e-4812-bf97-f1eb1afa41ce)


