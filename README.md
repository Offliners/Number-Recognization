# Number Recognization
recognize handwriting number with CNN model (dataset : MNIST)

### Directory
```
├─model
├─├─loss  // image of trinaing loss and validation loss
├─├─accuracy  // image of trinaing accuracy and validation accuracy
├─├─weight  // model weight
├─Image
├─├─True  // Identify handwriting number correctly
├─├─Wrong  // Identify handwriting number wrongly
```

### Usage
1. Run `main.py` to train model (also you can use the model weight that has trained)
2. Run `drawingBoard.py` to write a number
3. After writing the number, press enter
4. AI will recognize the number

### CNN Model
<div align=center><img width="900" height="1200" src=https://github.com/Offliners/Number-Recognization/blob/master/CNN_model.png/></div>

### Requirement
|Package|Version|
|-|-|
|Anaconda|`1.9.6`|
|Keras|`2.2.4`|
|Matplotlib|`3.0.3`|
|Numpy|`1.16.2`|
|Pygame|`1.9.6`|
|Tkinter|`8.6`|

### Reference
https://github.com/techwithtim/Number-Guesser-Neural-Net
