# Deep Model-Based Super-Resolution with Non-uniform Blur

This repository implements the code of *Deep Model-Based Super-Resolution with Non-uniform Blur*

# Train

To train the code please first download COCO dataset available at: https://cocodataset.org.

```
python main_train.py -opt options/train_nimbusr.json
```

# Test

Pre-trained model is available at: *model_zoo/NIMBUSR.pth*

See *test.ipynb* to test the model on COCO dataset.

# Results

We achieve state-of-the-art results in super-resolution in the presence of spatially-varying blur.
Here are some of the results we obtained. Feel free to test on your own sample using the testing notebook.

LR | SwinIR           |  BlindSR | USRNet | Ours | HR 
:-:|:------------------:|:-------:|:---:|:------:|:----:
<img src="images/Visual_res/kmap_1.png" alt="" width="400"/>  |  <img src="images/Visual_res/SwinIR_1.png" alt="" width="400"/> | <img src="images/Visual_res/BlindSR_1.png" alt="" width="400"/>  | <img src="images/Visual_res/USRNet_1.png" alt="" width="400"/>  | <img src="images/Visual_res/Ours_1.png" alt="" width="400"/>  | <img src="images/Visual_res/HR_1.png" alt="" width="400"/>
<img src="images/Visual_res/kmap_1_small.png" alt="" width="400"/>  |  <img src="images/Visual_res/SwinIR_1_small.png" alt="" width="400"/> | <img src="images/Visual_res/BlindSR_1_small.png" alt="" width="400"/>  | <img src="images/Visual_res/USRNet_1_small.png" alt="" width="400"/>  | <img src="images/Visual_res/Ours_1_small.png" alt="" width="400"/>  | <img src="images/Visual_res/HR_1_small.png" alt="" width="400"/>
<img src="images/Visual_res/kmap_2.png" alt="" width="400"/>  |  <img src="images/Visual_res/SwinIR_2.png" alt="" width="400"/> | <img src="images/Visual_res/BlindSR_2.png" alt="" width="400"/>  | <img src="images/Visual_res/USRNet_2.png" alt="" width="400"/>  | <img src="images/Visual_res/Ours_2.png" alt="" width="400"/>  | <img src="images/Visual_res/HR_2.png" alt="" width="400"/>
<img src="images/Visual_res/kmap_2_small.png" alt="" width="400"/>  |  <img src="images/Visual_res/SwinIR_2_small.png" alt="" width="400"/> | <img src="images/Visual_res/BlindSR_2_small.png" alt="" width="400"/>  | <img src="images/Visual_res/USRNet_2_small.png" alt="" width="400"/>  | <img src="images/Visual_res/Ours_2_small.png" alt="" width="400"/>  | <img src="images/Visual_res/HR_2_small.png" alt="" width="400"/>
<img src="images/Visual_res/kmap_3.png" alt="" width="400"/>  |  <img src="images/Visual_res/SwinIR_3.png" alt="" width="400"/> | <img src="images/Visual_res/BlindSR_3.png" alt="" width="400"/>  | <img src="images/Visual_res/USRNet_3.png" alt="" width="400"/>  | <img src="images/Visual_res/Ours_3.png" alt="" width="400"/>  | <img src="images/Visual_res/HR_3.png" alt="" width="400"/>
<img src="images/Visual_res/kmap_3_small.png" alt="" width="400"/>  |  <img src="images/Visual_res/SwinIR_3_small.png" alt="" width="400"/> | <img src="images/Visual_res/BlindSR_3_small.png" alt="" width="400"/>  | <img src="images/Visual_res/USRNet_3_small.png" alt="" width="400"/>  | <img src="images/Visual_res/Ours_3_small.png" alt="" width="400"/>  | <img src="images/Visual_res/HR_3_small.png" alt="" width="400"/>

LR | SwinIR           |  BlindSR | USRNet | Ours  
:-:|:------------------:|:-------:|:---:|:------:
<img src="images/Generalization/1_LR.png" alt="" width="400"/>  |  <img src="images/Generalization/1_SwinIR.png" alt="" width="400"/> | <img src="images/Generalization/1_BlindSR.png" alt="" width="400"/>  | <img src="images/Generalization/1_USRNet.png" alt="" width="400"/>  | <img src="images/Generalization/1_nimbusr.png" alt="" width="400"/>  
<img src="images/Generalization/1_small_LR.png" alt="" width="400"/>  |  <img src="images/Generalization/1_small_SwinIR.png" alt="" width="400"/> | <img src="images/Generalization/1_small_BlindSR.png" alt="" width="400"/>  | <img src="images/Generalization/1_small_USRNet.png" alt="" width="400"/>  | <img src="images/Generalization/1_small_nimbusr.png" alt="" width="400"/>  


# Acknowledgement
The codes use [KAIR](https://github.com/cszn/KAIR) as base. Please also follow their licenses. I would like to thank them for the amazing repository.