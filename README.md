# Face-Recognition-System-Robust-to-Identify-a-Covered-Face-using-Deep-Learning
## Teammates
- Jae Yoon Chung
- Kong Minsik
## Abstract
As an influence of the COVID 19 pandemic, everyone is required to wear a facial mask. This made it hard to identify the user’s face using the present facial recognition systems with portable devices. Using deep learning, our project resolves this problem by creating a system that identifies the user accurately and efficiently, even when the face is covered by an accessory.
## Algorithm Diagram
![그림1](https://user-images.githubusercontent.com/90415099/147800943-46cd4f1e-f71e-4ce0-9a83-a62ea35dea7f.png)
## Process
### 1. Accessory Detector
This model identifies the accessory the user is wearing. The model uses the MobileNet V2 architecture with a new head layer. The weights of architecyure is set as the wieghts of the ImageNet. Fine tuning, which freezes the convolutional base and training the classifier, is used to decide the weights of the head layer. 
![그림2](https://user-images.githubusercontent.com/90415099/147801268-ea5968d2-75d4-419a-9ac4-b3ad9731214f.png)
### 2. Face Recognizer 
The model used for facial recognition is trained with the images of the registered users. The 128-D embeddings of the images are extracted and used to decide the embeddings for each user. This is done by using the triplet loss function. By comparing the embeddings of the input image and the registered user, the model predicts who is in the input image.
![그림3](https://user-images.githubusercontent.com/90415099/147801667-e689abcc-f5a4-4cfb-91bb-0e22ffbc7960.png)
### 3. Image Composition
When face of the input image is covered by an accessory, the accuracy of the facial recognition is low. The image composition process resolves this problem by recreating the face of the input image into an uncovered face. The given graphs compares two situations: compositing images with the same user’s image and a different user’s image. The graphs show that compositing the image with the same user gives a higher accuracy.
![image](https://user-images.githubusercontent.com/90415099/147802038-62c9a6b8-3dac-4618-8a05-8feae03ce515.png)
![image](https://user-images.githubusercontent.com/90415099/147802052-d8fa1675-2219-4ee8-86a9-0b48fa783b8c.png)

## Code Instruction
### 1. Run accessory_detector/train_accessory_detector.py
Excecution code : python train_accessory_detector.py --dataset dataset_accessory<br />
The directory dataset_accessory has the datasets of default faces and faces with masks. You may add more images into each directories.<br />
If you want to add the type of accessories to detect, create a new directory and add images of faces wearing it.<br />
=> Output : accessory_detector.model
### 2. Go to dir output_creator
### 3. Download nn4.small2.v1.t7
[nn4.small2.v1.t7](http://cmusatyalab.github.io/openface/)
### 4. Run codes
- extract_embeddings_default.py<br />
Excecution Code : python codes/default/extract_embeddings_default.py --dataset dataset --embeddings output/embeddings_default.pickle --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7
- train_model_default.py<br />
Excecution Code : python codes/default/train_model_default.py --embeddings output/embeddings_default.pickle --recognizer output/recognizer_default.pickle --le output/le_default.pickle
### 4-1. Repeat step4 with mask
### 4-2. Test result
Test the outputs of step4 and step4-1 with cameras.<br />
Excecution Code : python codes/mask/recognize_video_mask.py --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7 --recognizer output/recognizer_mask.pickle --le output/le_mask.pickle

## Result
### - Primary Fcae Detector
![image](https://user-images.githubusercontent.com/90415099/147802102-633ba4e2-ba3b-42e9-9c5b-a54c8bc76150.png)

### - Secondary Face Detector 
![image](https://user-images.githubusercontent.com/90415099/147802087-f39ebebd-3e12-4eca-8f07-3d52540630dd.png)
