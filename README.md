# EfficientSwin
### **EfficientSwin: A Hybrid Model for Blood Cell Classification with saliency maps visualization**
![main figure](E_swin%20(1).png)
> **Abstract:** *Blood cell (BC) classification holds significant importance in medical diagnostics as it enables the identification and differentiation of various types of BCs, which is crucial for detecting specific infections, disorders, or conditions, and guiding appropriate treatment decisions. Accurate BC classification simplifies the evaluation of immune system performance and the diagnosis of various ailments such as infections, leukemia, and other hematological disorders. Deep learning algorithms perform excellently in the automated identification and differentiation of various types of BCs. One of the advanced deep learning models, EfficientNet has shown remarkable performance with limited datasets, another model Swin Transformer’s capability to capture intricate patterns and features makes it more accurate, albeit with limitations due to its large number of parameters. However, medical image datasets are often limited, necessitating a solution that balances accuracy and efficiency. To address this, we propose a novel hybrid model, EfficientSwin, which combines Swin Transformer’s and EfficientNet’s strengths. We first fine-tuned the Swin Transformer on a blood cell dataset comprising wihite blood cells, red blood cells and platelets, achieving promising outcomes. Subsequently, our hybrid model, EfficientSwin, outperformed the standalone Swin Transformer, achieving an impressive 98.14\% accuracy in blood cell classification. Furthermore, we compared our approach with previous research on white blood cell datasets, showcasing the superiority of EfficientSwin in accurately classifying blood cells.  Additionally, we utilized saliency maps to visually represent the classification results.* 
<hr />
## Installation
1. Install timm to load pretrained models
```shell
pip install timm
```
2. Install dataset bloodMNIST from medMNIST
```shell
pip install medmnist
```
3. Check available dataset
```shell
python -m medmnist available
```

<hr />

## Evaluation
Download the pretrained weights and run the following command for evaluation and training of proposed hybrid model on bloodMNIST dataset.
```shell
python EfficientSwin.py --batch_size 32 --epochs 35 --lr 0.001 --save_fig training_validation_accuracy.png
```
<hr />

## Comparison with Previous models on blood cell calssification
![results](images/table_2.png)

