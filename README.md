# enertiv-meter-reading
- Matthew Fielder matthew.fielder@colorado.edu
- Penglei Huang penglei.huang@olorado.edu
- Ethan Kellerhals ethan.kellerhals@colorado.edu

## Model
We have 2 types of models, Faster RCNN model and CNN model. Since we train it with different datasets, we have 5 model.pt files in total. The Faster RCNN model is used for region detection and the CNN model is for reading the meter value. **Generate those models first before running anything**

1. final_version/detectionModel.pt: ```python3 final_version/detection.py```
2. final_version/digitalModel.pt:  ``` python3 final_version/digitalDetection.py```
3. final_version/digitsModel.pt:  ``` python3 final_version/analogDetection.py```
4. final_version/cnn_model/analogreadingmodel.pt: ``` python3 final_version/cnn_model/analog_cnn.py```
5. final_version/cnn_model/digitalreadingmodel.pt: ``` python3 final_version/cnn_model/digital_cnn.py```

## Main program
We have our program in the folder **final_version** and it contain 2 user interfaces, Jupyter notebook and Command line UI.
### Jupyter notebook demo
We explain in detail our working process
![](/screenshots/notebook.png)
### Command line UI
```python3 run.py [image path] [image path]...```
![](/screenshots/cmd.png)

## Datasets
All the datasets are in the datasets folder and we had to annotate those images via drawing bounding box on the images in order to be used as training data. Here are the links of our datasets that we found. Some of our data came from our sponsor and images that our team took:
- http://artelab.dista.uninsubria.it/downloads/datasets/automatic_meter_reading/gas_meter_reading/gas_meter_reading.html
- https://www.kaggle.com/datasets/tapakah68/yandextoloka-water-meters-dataset
- https://github.com/SachaIZADI/Seven-Segment-OCR
- https://www.kaggle.com/datasets/frankhaverland/tenthofstepofmeterdigits
- https://www.kaggle.com/datasets/testtor/sevensegment-numbers
