# EvoluEarth

## Description
I've developed EvoluEarth because of global warming, urbanization, and insufficient agricultural areas. With artificial intelligence technologies, I aim to notice these developments and changes earlier before they happen. I suggest taking action to prevent them. Additionally, I will include comparisons of the state of these issues from 50 years ago to the present.

The project utilizes advanced machine learning techniques to predict the current state and potential future transformations of satellite images. By predicting how specific areas are likely to evolve, the system can provide actionable insights and recommendations to mitigate adverse impacts, fostering more informed decision-making for environmental conservation and urban planning.

I found this dataset from Kaggle. Name of dataset was "EuroSat Dataset". There are approximately 30,000 images. There are 10 classes (Annual Crop, Forest, Residential, River, Sealake etc. ) and each image has 64x64 pixels. We downloaded dataset from Kaggle and we distributed it to files.

## Image Preprocessing
Each pixel of each image scaled between 0 and 1 by dividing each pixels number with 255. I've used two types of augmentation. First of them is horizontal flip and the second one is vertical flip. I seperated %25 of dataset as validation and rest of them are training data.
![image](https://github.com/denizbilgin/EvoluEarth/blob/main/imgs/augmented_images.png)

## Training
I've trained lots of models, you can see statistics of some training processes.
![image](https://github.com/denizbilgin/EvoluEarth/blob/main/imgs/training_statistics.png)

The experiments with the highest validation accuracy (0.9689) used a learning rate of 4e-4, the ReLU activation function, and the Adamax optimizer. Epochs numbers are 38.
The experiment with the shortest training time (25 minutes) used a learning rate of 3e-3, the SeLU activation function, and the Adam optimizer, but it also had the lowest validation accuracy (0.75).
Overall, the table suggests that the best hyperparameter settings for this task are a learning rate of 4e-4, the ReLU activation function, and the Adamax optimizer.

![image](https://github.com/denizbilgin/EvoluEarth/blob/main/imgs/losses.png)
![image](https://github.com/denizbilgin/EvoluEarth/blob/main/imgs/losses2.png)
![image](https://github.com/denizbilgin/EvoluEarth/blob/main/imgs/accuracies.png)

When I trained models which is deeper and has more neurons, it became worse and worse. Models which are more shallow did better performance than deepers.

## Final Tests
![image](https://github.com/denizbilgin/EvoluEarth/blob/main/imgs/amazon.png)
Output of the model for this evolution:

Forest areas were converted into highways.

Neutral habitats are destroyed and biodiversity is decreased.

Deforestation contributes to global warming by reducing carbon sequestration.

Eco-friendly transportation solutions should be developed.

![image](https://github.com/denizbilgin/EvoluEarth/blob/main/imgs/levent.png)
Output of the model for this evolution:

Forest areas were converted into residential zones.

Natural habitats are destroyed and biodiversity is decreased.

Deforestation contributes to global warming by reducing carbon sequestration.

Sustainable urban planning should be implemented.

![image](https://github.com/denizbilgin/EvoluEarth/blob/main/imgs/adana.png)
Annual crop areas remained unchanged. The area has been protected and no significant changes occurred.

## Confusion Matrix of the Best Model
![image](https://github.com/denizbilgin/EvoluEarth/blob/main/imgs/cm.png)
