my_config = {
    "program": "main.py",
    "method": "grid",
    "name": "Sweep",
    "parameters":
    {
        "random_seed":
            {"value": 21},
        "dataset_path":
            {"value": '/opt/ml/segmentation/input/data'},
        "train_path":
            {"value": '/opt/ml/segmentation/input/data/train.json'},
        "val_path":
            {"value": '/opt/ml/segmentation/input/data/val.json'},
        "test_path":
            {"value": '/opt/ml/segmentation/input/data/test.json'},
        "dataset":
            {"value": 'CustomDataLoader'},
        "category_names":
            {"value": ('Backgroud', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing')},
        # "augmentation":
        #     {"value": 'my_transform'},
        # "resize":
        #     {"value": (256, 192)},
        "optimizer":
            {"values": ['sgd', 'momentum', 'adam']},
        "model":
            {"values": ['fcn_resnet50', 'fcn_resnet101']},
        "batch_size":
            {"values": [8, 16]},
        "lr":
            {"values": [0.0001]},
        "lr_decay_step":
            {"value": 10},
        "criterion":
            {"values": ['cross_entropy']}, #, 'focal', 'label_smoothing', 'F1Loss']},
        "num_epochs":
            {"value": 3},
        "scheduler":
            {"value": 'steplr'},
        "saved_dir":
            {"value": './saved'},
        "val_every":
            {"value": 1},
        "name":
            {"value": 'sweep'},
    }
}