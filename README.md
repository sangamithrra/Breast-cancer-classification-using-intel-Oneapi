# Breast-cancer-classification-using-intel-Oneapi

The objective of this project is to develop a classification model using KNN and logistic regression algorithms to classify tumor cells as malignant or benign based on clinical and pathological features. The early detection and diagnosis of breast cancer are essential for effective treatment and survival. Machine learning algorithms, such as k-nearest neighbors (KNN) and logistic regression, can be used to develop accurate classification models to aid in the diagnosis of breast cancer The model aims to achieve high accuracy, sensitivity, and specificity to assist healthcare providers in making informed decisions about patient care.

![image](https://user-images.githubusercontent.com/113981140/235284876-88740033-05d8-4e62-80bf-916cfb7b9398.png)

Problem Statement:

The problem statement for the project is to develop a machine learning classification model that accurately classifies breast cancer cases as malignant or benign based on clinical and pathological features. Early detection and diagnosis of breast cancer are crucial for effective treatment and survival. The current diagnostic methods are not always accurate, and there is a need for more reliable and efficient diagnostic tools. The objective of this project is to address this problem by developing a classification model using KNN and logistic regression algorithms that achieves high accuracy, sensitivity, and specificity. This model can assist healthcare providers in making informed decisions about patient care, leading to earlier detection, timely treatment, and improved patient outcomes.

Solution:

The proposed solution for classifying breast tumor cells into malignant or benign involves the use of machine learning algorithms like k-nearest neighbors and logistic regression. 
We use Intel one API to optimize and accelerate the implementation of these algorithms, allowing for efficient processing of large datasets. The solution involves a multi-stage process that includes data preprocessing, feature selection, model training, and evaluation. 
The performance of the models will be evaluated using metrics such as accuracy, precision, recall, and F1 score. The final solution will provide accurate and reliable predictions for breast cancer diagnosis, improving the effectiveness of medical treatments and reducing the number of misdiagnoses


WHY ONEAPI

![image](https://user-images.githubusercontent.com/113981140/235285065-b08846f7-4326-4d2f-8299-c95e4f5c22a6.png)

Intel One API is a set of developer tools designed to accelerate workloads across CPUs, GPUs, and other accelerators. It provides a unified programming model that simplifies the development of high-performance applications and allows developers to take advantage of hardware acceleration to speed up their workloads.

In a machine learning project, Intel One API can be used to optimize the performance of training and inference of machine learning models on a range of hardware platforms. This can lead to faster model training times, lower latency in making predictions, and improved overall performance.

Additionally, Intel One API provides a range of pre-optimized machine learning primitives and libraries that can be used to accelerate common machine learning operations such as convolution, matrix multiplication, and activation functions. This can further improve the performance of machine learning models and reduce the time required for development and optimization.

Toolkit used: Intel® AI Analytics Toolkit (AI Kit)

The Intel® AI Analytics Toolkit (AI Kit) helps to resolve this problem by providing better results by optimising the models.
Scikit-learn (sklearn) is a popular Python library for machine learning, but it may not be optimized for Intel hardware by default.By integrating oneAPI into scikit-learn, the performance of some of its algorithms, such as linear regression, can be significantly improved.integrating oneAPI into scikit-learn can improve its performance on Intel hardware by taking advantage of highly optimized libraries and routines for mathematical computations and data analysis.

![image](https://user-images.githubusercontent.com/113981140/235285147-d3699f6c-4e3d-4017-8711-78237b327eab.png)

Intel DevCloud
Link: https://jupyter.oneapi.devcloud.intel.com/user/u190404/doc/tree/Breast%20cancer%20classification.ipynb

Results and discussion

The accuracy using KNeighborsClassifier is 95.104895.

The oneAPI reduced the overall runtime and GPU usage significantly compared to normal platforms. All the models haven been optimised and execution time have been reduced by using Intel oneAPI.The execution time was reduced 26 times compared to other platform.

![homepage](https://user-images.githubusercontent.com/113981140/235285238-921c7030-9916-47c0-bd43-2dad3b0ccd79.png)

![featured image](https://user-images.githubusercontent.com/113981140/235285247-59872c4d-2f22-4d73-860c-93726b3dbf6b.png)

Web app implementation vedio:

https://github.com/sangamithrra/Breast-cancer-classification-using-intel-Oneapi/blob/main/Cancer%20Tumor%20Classification%20-%20Google%20Chrome%202023-04-29%2011-39-38.mp4









