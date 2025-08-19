# Enhanced PPE Detection in Low-Light Tunnel Environments: A YOLOv5-Based Approach

## 1. Introduction

Addressing the challenges of personal protective equipment (PPE) detection in low-light tunnel environments, this paper introduces an improved YOLOv5-based method. The approach integrates a Channel-Metric (CM) attention mechanism, an Adaptive Feature Pyramid Network (AFPN), and an XIoU_NMS function to enhance detection robustness, small target detection, and occluded target detection. Experimental results demonstrate significant improvements, with a detection accuracy of 94.6% and a 15% increase in small target recall. The model's stable performance in real tunnel monitoring systems underscores its potential for enhancing construction safety management.
![image1](图片1.png)
---

## 2. Requirements

### 2.1 Hardware Requirements
- GPU: NVIDIA RTX 4090 (20GB) or equivalent  
- CPU: Intel i9 or higher  
- RAM: ≥ 32GB  

### 2.2 Software Requirements
- OS: Ubuntu 20.04 / Windows 10  
- CUDA: 11.3  
- cuDNN: 8.2  

### 2.3 Python Dependencies
```bash
conda create -n ppe_yolov5 python=3.8 -y
conda activate ppe_yolov5
pip install -r requirements.txt
````

---

## 3. Training

### 3.1 CM Attention Module

```python
class CM_Attention(nn.Module):
    def __init__(self, channels):
        super(CM_Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 16, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
```
![image2](图片2.png)

### 3.2 AFPN

The AFPN structure is defined in the `afpn.yaml` configuration file. Running `afpn.py` will instantiate and execute the model:

```bash
# Load configuration and run AFPN
python afpn.py --cfg afpn.yaml
```


### 3.3 XIoU\_NMS Function

```python
def xiou_nms(boxes, scores, threshold=0.5):
    # Replace IoU with XIoU for suppression, enhancing performance under occlusion
    keep = []
    # Implementation is similar to standard NMS but uses XIoU calculation
    return keep
```

---

## 4. Experiments

### 4.1 Dataset

* Source: Tunnel construction monitoring videos (infrared / low-light conditions)
* Classes: PPE (helmet, reflective vest, protective glasses, etc.)
* Scale: 10,000+ annotated images

### 4.2 Comparative Results

| Model                                       | Precision | Recall | mAP   | F1    |
|--------------------------------------------|-----------|--------|-------|-------|
| Faster-RCNN                                 | 0.761     | 0.603  | 0.645 | 0.673 |
| SSD300                                      | 0.814     | 0.722  | 0.754 | 0.765 |
| YOLOv3                                      | 0.829     | 0.703  | 0.744 | 0.761 |
| YOLOv4                                      | 0.740     | 0.623  | 0.663 | 0.676 |
| YOLOv8                                      | 0.908     | 0.831  | 0.842 | 0.868 |
| Transformer Mutual Attention  | 0.915     | 0.840  | 0.850 | 0.877 |
| EAPT                     | 0.920     | 0.845  | 0.855 | 0.880 |
| **Ours**                                    | 0.946     | 0.883  | 0.902 | 0.913 |

![image3](图片3.png)

---

## 5. Conclusion

This project introduces improvements including **CM attention mechanism, AFPN multi-scale fusion, and XIoU\_NMS suppression**, which significantly enhance detection accuracy and robustness for PPE detection in low-light tunnel environments.

Future work will focus on:

* Multi-modal fusion (infrared + visible light)
* Temporal information integration for video detection
* Broader application in diverse construction safety scenarios

---




