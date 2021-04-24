# WMH_Segmentation
White matter hyperintensity segmentation to find lesions in the brain. 

WMH Segmentation using data available on https://wmh.isi.uu.nl/

Using T1 and FLAIR images for predictions. 

40 subjects split in 8:2 train to test split used. First sliced and normalized across volume and trained on 2D slices.

Labels 0, 1, 2 corresponding to background, WMH and other pathologies, respectively. Label to multi-channel transform used to create 2 channel representing WMH and background(merging class 0 and 2). 

## **Model:**
UNet(

    dimensions=2,
    in_channels=2,    
    out_channels=2,    
    channels=(16, 32, 64, 128, 256),    
    strides=(2, 2, 2, 2),    
    num_res_units=2,        
    dropout=0.2,    
    kernel_size=3,
)

## **Optimizer:**
optimizer = torch.optim.Adam(
    model.parameters(), 1e-3, weight_decay=1e-5, amsgrad=True, 
)

## **Sample training Predictions at various epochs (Output=Predictions):**

**Image Channel 0 - T1**

**Image Channel 1 - FLAIR**
Epoch 42

![image](https://user-images.githubusercontent.com/43177212/115964885-6bbbbe00-a51e-11eb-8a0d-88f7785cbbb9.png)
![image](https://user-images.githubusercontent.com/43177212/115964889-6e1e1800-a51e-11eb-87f1-77ed2574ebd5.png)
![image](https://user-images.githubusercontent.com/43177212/115964890-6fe7db80-a51e-11eb-8c7c-e35fef4c83c4.png)
![image](https://user-images.githubusercontent.com/43177212/115964895-724a3580-a51e-11eb-8027-49d592be2832.png)

Epoch 334

![image](https://user-images.githubusercontent.com/43177212/115964967-d836bd00-a51e-11eb-9334-78cc3336076d.png)
![image](https://user-images.githubusercontent.com/43177212/115964972-db31ad80-a51e-11eb-8e22-77ae167bdbc2.png)
![image](https://user-images.githubusercontent.com/43177212/115964974-dcfb7100-a51e-11eb-9e71-ebf066538eb5.png)
![image](https://user-images.githubusercontent.com/43177212/115964975-df5dcb00-a51e-11eb-909d-0de1cdd4e659.png)


## **Training and validation loss: **

![image](https://user-images.githubusercontent.com/43177212/115964730-b6890600-a51d-11eb-9f65-1ffa0ee043c4.png)

## **Prediction:**
8 subject WMH segmentation DICE score
[0.9567131996154785, 0.9833950400352478, 0.9731700420379639, 0.9664475917816162, 0.9873022437095642, 0.9588155746459961, 0.978124737739563, 0.954979658126831]

Mean DICE score
0.96987 (5 d.p.)

**Hausdorff distance score:**
To implement

**Sample subject prediction visualization:**
![image](https://user-images.githubusercontent.com/43177212/115964521-cc49fb80-a51c-11eb-86a9-c664705cb316.png)


## **Tasks to do:**
- Implement scheduler to improve loss by reducing learning rate over epoch
- Move code into Project folder and separate into relevante modules and classes
- Train on more data (if applicable and relavant for segmentation task in question)
- Predict on more dataset and compare results. 
