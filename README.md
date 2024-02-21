# Cric (Cytopathalogy)


path to dataset :- https://drive.google.com/drive/folders/1UARjzLLCFcfRscnQ5hoIwH6c_CDrepxQ?usp=share_link

split Train-Val-Test 
```
python Train_val_test_split.py --dataset='path to ur csv label file' 
```

Train ResNet50
```
python main.py --num_classes=2 --num_epochs=100 --img_dir='/path to img dir' --model='CustomResNet'

```

Train DieT
```
python main.py --num_classes=2 --num_epochs=100 --img_dir='/path to img dir' --model='Custom_ViT'

```


