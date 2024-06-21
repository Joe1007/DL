DeepLaerning HW2
===
1.Envirnoment
---
The Python version we use is `python 3.11.9`, and other imortant packages like `torch`, `torchvision`, `scikit-learn`, we jsued use the `pip install <packagename>`to get the default.

2.Run the code
---
Download the [dataset](https://drive.google.com/drive/u/0/folders/12zWyLBtxYiMdJrNj6ZnWxlTqGbwWyBbH), and unzip them. You should keep the folder `images`, files `train.txt`, `val.txt`, `test.txt` with the `.ipynb` under the same path.  
Download the [dcheckpoints](https://drive.google.com/drive/u/0/folders/1JniaonG4AznSNXTkdxKcWO6NYSc3MFa6), still keep them with the `.ipynb` under the same path.  
After putting the dataset and weights under the correct path, you run the `.ipynb`, should get the inference results immediately.

### Note
Different `.ipynb`use different `checkpoints`, we list as following:
1. `HW2_ResNet34.ipynb`--`checkpoint.pth`
2. `Visualizedloss_epoch50.ipynb`--`checkpoint_cnn_rrdb.pth`
3. `HW2_Dynamic_RDBB`--`checkpoint_cnn_rrdb1.pth`, `checkpoint_cnn_rrdb1.pth`  
   (when use `checkpoint_cnn_rrdb1.pth`, you must keep the row `self.rrdb1 = RRDB(64, 32)`, the RRDB channel is `32`  
    use  `checkpoint_cnn_rrdb2.pth`, you must keep the row `self.rrdb1 = RRDB(64, 64)`, the RRDB channel is `64`)
```
# 定義CustomCNN_DynamicConv2d_RRDB模型
class CustomCNN_DynamicConv2d_RRDB(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN_DynamicConv2d_RRDB, self).__init__()
        self.dynamic_conv = DynamicConv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.rrdb1 = RRDB(64, 32)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128 * 32 * 32, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.dynamic_conv(x)))
        x = self.rrdb1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x
```
4. `HW2_Dynamic(noRDBB).ipynb`--`checkpoint_cnn_rrdb3.pth`
5. `SpecialConV.ipynb`--`checkpoint_cnn_R.pth`, `checkpoint_cnn_RG.pth`, `checkpoint_cnn_RGB.pth`
