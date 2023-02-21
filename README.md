# GatingAno

This is an official pytorch implementation of 'Anomaly Detection via Gating Highway Connection for Retinal Fundus Images'. 

![Method](./teaser.png)

## Requirements

*   numpy>=1.17.0
*   scipy>=1.5.2 
*   Pillow>=8.2.0
*   pytorch>=1.7.1
*   torchvision>=0.8.2
*   tqdm>=4.59.0
*   scikit-learn>= 0.24.2
*   scikit-image>=0.17.2

## Datasets 
The proposed method is evaluated on two publicly-available datasets, i.e. 

*   [IDRiD](https://www.sciencedirect.com/science/article/pii/S1361841519301033?casa_token=pO8u1MuAw1wAAAAA:Yx2KH3-xTfilsRS7Q_Nafrl3RgIeC4rMSuod14mlVWNOUF8OqD_THuZDaJglEsqJ2GfMUEhrO992)
*   [ADAM](https://ieeexplore.ieee.org/abstract/document/9768802)

## Usage
The proposed GatingAno method is trained through two steps:
*   Data Preparation
    
    Generate the list of HOG image and Patches :
    ```
    python3 data_find.py \
    --dataset ['IDRiD'/'IDRiDc'/'ADAM'/'ADAMc'] \
    --path {data dir}
    ```
    
    For example, to generate the image-level label of IDRiD dataset, you can run
    `python3 data_find.py --dataset 'IDRiDc' --path './dataset/' `
    
    And then you will get lists containing images and corresponding labels in './label/'. 
    
*   Training and testing model
    
    For example, to train pixel-level anomaly detection task on ADAM dataset, you can run
     ```
     python3 main.py \
     --dataset 'ADAM' \
     --datadir './labels/ADAM/' \
     --lr 1e-3 \
     --level 'pixel' ;
     ```
