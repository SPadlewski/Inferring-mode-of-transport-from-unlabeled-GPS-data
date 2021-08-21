
# Inferring modes of transportation from raw unlabelled GPS data using Convolutional Neural Networks

## Project's Abstract

The comprehension of the peoples' transportation patterns, travel behaviours and the modal split of individuals' journeys are essential insights used for demand analysis, traffic management, or optimisation of travel time. In this project, the user's modes of transport are inferred using data from everyday devices with GPS capabilities, ubiquitous in the modern world, through the utilisation of a convolutional neural network (CNN). Moreover, this work enhances the already established techniques relying solely on motion features extracted from GPS data for training the model by proposing the inclusion of proximity features, calculated using transportation networks characteristics points' locations. The CNN models are trained on a separate set of features which predictions are inserted in the logistic regression model to derive the ultimate classification. Such an approach is based on an ensemble technique called stacked generalisation and proved to be beneficial for the inferring process strengthening the accuracy level by 1.1 per cent compared to the latter approach.
This dissertation also creates a unique methodology that strives to help pave the path for more real-world application of the existing state-of-the-art classification methods. It proposes that by testing whether users' modes of transport can be reliably identified from raw unlabelled GPS data gathered using a heterogeneous set of GPS devices and from a dissimilar region than the dataset used for training the models.
Although the classification accuracy values obtained in this dissertation haven't been sufficient to prove the truthfulness of our core hypothesis, they exhibit a notable potential of the proposed approach. Additionally, this dissertation, with its unique methodology framework, can be seen as a proof of concept moving the knowledge base forward.

## Project's Methodology 
![method-01](https://user-images.githubusercontent.com/72401277/130332631-a79a0c31-5b52-4cbe-b66a-9dd9ec74a25c.png)

### Phase 1
<br>1.Labeled_CNN_pre_processing.ipynb </br>
<br>2.Labeled_CNN_input_Layer_Motion.ipynb </br>
<br>3.Labeled_CNN_input_Layer_Proximity.ipynb </br>
<br>4.Labeled_CNN_Keras_Data_Creation_Proximity.ipynb </br>
<br>5.Labeled_CNN_Keras_Data_Creation.ipynb </br>
<br>6.Labeled_CNN_Keras_Data_Creation_Proximity_For_Stacking.ipynb </br>
<br>7.Labeled_CNN_Model_Motion.ipynb </br>
<br>8.Labeled_CNN_Model_Proximity.ipynb </br>
<br>9.Labeled_CNN_ensemble_motion.ipynb </br>
<br>10.Labeled_CNN_ensemble_proximity.ipynb </br>
<br>11.Labeled_CNN_Stacking.ipynb</br>

### Phase 2
<br>1.Unlabeled_CNN_pre_proccesing.ipynb</br>
<br>2.Unlabeled_CNN_input_layer.ipynb </br>
<br>3.Unlabeled_CNN_input_layer_averages.ipynb </br>

### Phase 3
<br>1.Unlabeled_CNN_classification-5.ipynb   </br>
<br>2.Unlabeled_CNN_classification-20.ipynb</br>
<br>3.Unlabeled_CNN_classification-40.ipynb</br>
<br>4.Unlabeled_CNN_classification-60.ipynb </br>
