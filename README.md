# Fashion_Image_Classification_Project_v2
The goal of this project was to create a multi-label image classificaion algorithm that can detect both the colour and the category of an article of clothing in an image.

In order to solve this problem I built out an algorithm in keras usind SmallerVGGNet neural network structure. This is a simplified version of the VGGNet model was first introduced by Simonyan and Zisserman in their 2014 paper, Very Deep Convolutional Networks for Large Scale Image Recognition.

I then trained this network in a supervised learning fashion using approximately 2,200 images which I had scraped from the web. The network is only trained to recognize the following categories and colours:

Category: Jeans, Dress, Shirt

Colour: Red, Blue, Black

The results from running the classify.py script on unseen fashion image examples can be seen below:


![Screen Shot 2021-02-24 at 20 38 56](https://user-images.githubusercontent.com/71552393/109056191-6ab60e00-76e0-11eb-8e28-322b7db1e1c9.png)

![Screen Shot 2021-02-24 at 20 50 23](https://user-images.githubusercontent.com/71552393/109057444-06944980-76e2-11eb-930e-55c29eff001e.png)
