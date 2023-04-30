# enertiv-meter-reading
- Matthew Fielder matthew.fielder@colorado.edu
- Penglei Huang penglei.huang@olorado.edu
- Ethan Kellerhals ethan.kellerhals@colorado.edu

## Model
We have 2 types of models, faster RCNN model and CNN model. Since we train it different datasets, we have 5 model.pt in total. Faster RCNN model for region detection and CNN model for reading the meter value. **Generate those models first before running anything**

1. final_version/detectionModel.pt: ```python3 final_version/detection.py```
2. final_version/digitalModel.pt:  ``` python3 final_version/digitalDetection.py```
3. final_version/digitsModel.pt:  ``` python3 final_version/analogDetection.py```
4. final_version/cnn_model/analogreadingmodel.pt: ``` python3 final_version/cnn_model/analog_cnn.py```
5. final_version/cnn_model/digitalreadingmodel.pt: ``` python3 final_version/cnn_model/digital_cnn.py```

## Main program
We have our program in the folder **final_version** and it contain 2 user interfaces, Jupyter notebook and Command line UI.
### Jupyter notebook demo
We explain in detail of our working process
![](/screenshots/notebook.png)
### Command line UI
```python3 run.py [image path] [image path]...```
![](/screenshots/cmd.png)

## Datasets
All the datasets in the datasets folder and we have to annotate those images via drawing bounding box in order to be as train data. Here is the links of our datasets:
- http://artelab.dista.uninsubria.it/downloads/datasets/automatic_meter_reading/gas_meter_reading/gas_meter_reading.html
- https://www.kaggle.com/datasets/tapakah68/yandextoloka-water-meters-dataset
- https://github.com/SachaIZADI/Seven-Segment-OCR
- https://www.kaggle.com/datasets/frankhaverland/tenthofstepofmeterdigits
- https://www.kaggle.com/datasets/testtor/sevensegment-numbers
