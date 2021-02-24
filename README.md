# Fashion_Image_Classification_Project_v2
The goal of this project was to create a multi-label image classificaion algorithm that can detect both the colour and category of clothing article present in an image.

In order to solve this problem I built out an algorithm in keras usind SmallerVGGNet neural network structure. This is a simplified version of the VGGNet model was first introduced by Simonyan and Zisserman in their 2014 paper, Very Deep Convolutional Networks for Large Scale Image Recognition.

I then trained this network in a supervised learning fashion using approximately 2,200 images which I had scraped from the web.

The results from running the classify.py script on unseen fashion image examples can be seen below:


![Screen Shot 2021-02-24 at 20 38 56](https://user-images.githubusercontent.com/71552393/109056191-6ab60e00-76e0-11eb-8e28-322b7db1e1c9.png)

![Screen Shot 2021-02-24 at 20 35 28](https://user-images.githubusercontent.com/71552393/109055905-0dba5800-76e0-11eb-9304-46e592f9f2db.png)
