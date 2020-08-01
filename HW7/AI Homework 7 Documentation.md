# AI Homework 7 Documentation


## File Structure
```
├───{extracted folder name}
│    ├───imgnet500
│    │    └───imagespart
│    │         ├───ILSVRC2012_val_00000001.JPEG
│    │         ├───ILSVRC2012_val_00000002.JPEG
│    │         ├───...
│    │         └───ILSVRC2012_val_00000500.JPEG
│    ├───hwhelpers
│    │    ├───utils
│    │    │    ├───imgnetdatastuff.py
│    │    │    └───guidedbpcodehelpers.py
│    │    ├───val
│    │    │    └───(xml files)
│    │    └───synset_words.txt
│    ├───vgg16
│    │    ├───ILSVRC2012_val_00000001_1
│    │    ├───ILSVRC2012_val_00000002_1
│    │    ├───ILSVRC2012_val_00000003_1
│    │    ├───...
│    │    └───ILSVRC2012_val_00000001_1
│    ├───vgg16_bn
│    │    ├───ILSVRC2012_val_00000001_1
│    │    ├───ILSVRC2012_val_00000002_1
│    │    ├───ILSVRC2012_val_00000003_1
│    │    ├───...
│    │    └───ILSVRC2012_val_00000001_1
│    ├───part1_main.py
│    ├───part2_main.py
```
## Instructions to Run

### Part 1
For Linux systems:
```
python3 part1_main.py
```
For Windows systems:
```
python part1_main.py
```

### Part 2
For Linux systems:
```
python3 part2_main.py
```
For Windows systems:
```
python part2_main.py
```

## Summary

Part 1 will output the 5th - 95th percentile of the weights

Part 2 will output the feature map