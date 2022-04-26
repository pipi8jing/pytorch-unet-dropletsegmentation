# pytorch-unet-dropletsegmentation
![image](https://user-images.githubusercontent.com/66905900/165304647-924ab673-650b-4ec9-96de-cce51a154473.png)
You might put the dir name(no superior path) of your train data, train label, validation data and validation label here(in fnet->data->tiffdataset.py) in order.
![image](https://user-images.githubusercontent.com/66905900/165305390-7e4629df-6764-4af4-b50a-58f7f808b66c.png)
You might put the path of your whole dataset here.(in config->train.json)
![image](https://user-images.githubusercontent.com/66905900/165306795-f84967a9-c4a2-4a3e-9a3d-d05da8a353fb.png)
You might put the path of test dataset, the path of model and the path of result here in order.(in config->predict.json)
![image](https://user-images.githubusercontent.com/66905900/165307697-f7df264e-8079-403f-908f-1851e4650163.png)
You might put the path of prediction and the path of label here in order to count dice.(in dice.py)
