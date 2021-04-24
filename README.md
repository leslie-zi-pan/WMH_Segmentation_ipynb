# WMH_Segmentation
WMH Segmentation using data available on https://wmh.isi.uu.nl/

Using T1 and FLAIR images for predictions. 

Labels 0, 1, 2 corresponding to background, WMH and other pathologies. Label to multi-channel transform used to create 2 channel representing WMH and background(merging class 0 and 2). 

Model:
UNet(
    dimensions=2,
    in_channels=2,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    dropout=0.2,
    kernel_size=3,
).to(device)

Otimizer: 
optimizer = torch.optim.Adam(
    model.parameters(), 1e-3, weight_decay=1e-5, amsgrad=True, 
)

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 16, 128, 128]             304
            Conv2d-2         [-1, 16, 128, 128]             304
    InstanceNorm2d-3         [-1, 16, 128, 128]               0
           Dropout-4         [-1, 16, 128, 128]               0
             PReLU-5         [-1, 16, 128, 128]               1
            Conv2d-6         [-1, 16, 128, 128]           2,320
    InstanceNorm2d-7         [-1, 16, 128, 128]               0
           Dropout-8         [-1, 16, 128, 128]               0
             PReLU-9         [-1, 16, 128, 128]               1
     ResidualUnit-10         [-1, 16, 128, 128]               0
           Conv2d-11           [-1, 32, 64, 64]           4,640
           Conv2d-12           [-1, 32, 64, 64]           4,640
   InstanceNorm2d-13           [-1, 32, 64, 64]               0
          Dropout-14           [-1, 32, 64, 64]               0
            PReLU-15           [-1, 32, 64, 64]               1
           Conv2d-16           [-1, 32, 64, 64]           9,248
   InstanceNorm2d-17           [-1, 32, 64, 64]               0
          Dropout-18           [-1, 32, 64, 64]               0
            PReLU-19           [-1, 32, 64, 64]               1
     ResidualUnit-20           [-1, 32, 64, 64]               0
           Conv2d-21           [-1, 64, 32, 32]          18,496
           Conv2d-22           [-1, 64, 32, 32]          18,496
   InstanceNorm2d-23           [-1, 64, 32, 32]               0
          Dropout-24           [-1, 64, 32, 32]               0
            PReLU-25           [-1, 64, 32, 32]               1
           Conv2d-26           [-1, 64, 32, 32]          36,928
   InstanceNorm2d-27           [-1, 64, 32, 32]               0
          Dropout-28           [-1, 64, 32, 32]               0
            PReLU-29           [-1, 64, 32, 32]               1
     ResidualUnit-30           [-1, 64, 32, 32]               0
           Conv2d-31          [-1, 128, 16, 16]          73,856
           Conv2d-32          [-1, 128, 16, 16]          73,856
   InstanceNorm2d-33          [-1, 128, 16, 16]               0
          Dropout-34          [-1, 128, 16, 16]               0
            PReLU-35          [-1, 128, 16, 16]               1
           Conv2d-36          [-1, 128, 16, 16]         147,584
   InstanceNorm2d-37          [-1, 128, 16, 16]               0
          Dropout-38          [-1, 128, 16, 16]               0
            PReLU-39          [-1, 128, 16, 16]               1
     ResidualUnit-40          [-1, 128, 16, 16]               0
           Conv2d-41          [-1, 256, 16, 16]          33,024
           Conv2d-42          [-1, 256, 16, 16]         295,168
   InstanceNorm2d-43          [-1, 256, 16, 16]               0
          Dropout-44          [-1, 256, 16, 16]               0
            PReLU-45          [-1, 256, 16, 16]               1
           Conv2d-46          [-1, 256, 16, 16]         590,080
   InstanceNorm2d-47          [-1, 256, 16, 16]               0
          Dropout-48          [-1, 256, 16, 16]               0
            PReLU-49          [-1, 256, 16, 16]               1
     ResidualUnit-50          [-1, 256, 16, 16]               0
   SkipConnection-51          [-1, 384, 16, 16]               0
  ConvTranspose2d-52           [-1, 64, 32, 32]         221,248
   InstanceNorm2d-53           [-1, 64, 32, 32]               0
          Dropout-54           [-1, 64, 32, 32]               0
            PReLU-55           [-1, 64, 32, 32]               1
         Identity-56           [-1, 64, 32, 32]               0
           Conv2d-57           [-1, 64, 32, 32]          36,928
   InstanceNorm2d-58           [-1, 64, 32, 32]               0
          Dropout-59           [-1, 64, 32, 32]               0
            PReLU-60           [-1, 64, 32, 32]               1
     ResidualUnit-61           [-1, 64, 32, 32]               0
   SkipConnection-62          [-1, 128, 32, 32]               0
  ConvTranspose2d-63           [-1, 32, 64, 64]          36,896
   InstanceNorm2d-64           [-1, 32, 64, 64]               0
          Dropout-65           [-1, 32, 64, 64]               0
            PReLU-66           [-1, 32, 64, 64]               1
         Identity-67           [-1, 32, 64, 64]               0
           Conv2d-68           [-1, 32, 64, 64]           9,248
   InstanceNorm2d-69           [-1, 32, 64, 64]               0
          Dropout-70           [-1, 32, 64, 64]               0
            PReLU-71           [-1, 32, 64, 64]               1
     ResidualUnit-72           [-1, 32, 64, 64]               0
   SkipConnection-73           [-1, 64, 64, 64]               0
  ConvTranspose2d-74         [-1, 16, 128, 128]           9,232
   InstanceNorm2d-75         [-1, 16, 128, 128]               0
          Dropout-76         [-1, 16, 128, 128]               0
            PReLU-77         [-1, 16, 128, 128]               1
         Identity-78         [-1, 16, 128, 128]               0
           Conv2d-79         [-1, 16, 128, 128]           2,320
   InstanceNorm2d-80         [-1, 16, 128, 128]               0
          Dropout-81         [-1, 16, 128, 128]               0
            PReLU-82         [-1, 16, 128, 128]               1
     ResidualUnit-83         [-1, 16, 128, 128]               0
   SkipConnection-84         [-1, 32, 128, 128]               0
  ConvTranspose2d-85          [-1, 2, 256, 256]             578
   InstanceNorm2d-86          [-1, 2, 256, 256]               0
          Dropout-87          [-1, 2, 256, 256]               0
            PReLU-88          [-1, 2, 256, 256]               1
         Identity-89          [-1, 2, 256, 256]               0
           Conv2d-90          [-1, 2, 256, 256]              38
     ResidualUnit-91          [-1, 2, 256, 256]               0
================================================================
Total params: 1,625,449
Trainable params: 1,625,449
Non-trainable params: 0

Training and validation loss: 
![image](https://user-images.githubusercontent.com/43177212/115964499-b1778700-a51c-11eb-9630-504bd3f02bf7.png)

Prediction:
8 subject WMH segmentation DICE score
[0.9567131996154785, 0.9833950400352478, 0.9731700420379639, 0.9664475917816162, 0.9873022437095642, 0.9588155746459961, 0.978124737739563, 0.954979658126831]

Mean DICE score
0.96987 (5 d.p.)

Hausdorff distance score:
To implement

Sample subject prediction visualization:
![image](https://user-images.githubusercontent.com/43177212/115964521-cc49fb80-a51c-11eb-86a9-c664705cb316.png)


Tasks to do:
- Implement scheduler to improve loss by reducing learning rate over epoch
- Move code into Project folder and separate into relevante modules and classes
- Train on more data (if applicable and relavant for segmentation task in question)
- Predict on more dataset and compare results. 
