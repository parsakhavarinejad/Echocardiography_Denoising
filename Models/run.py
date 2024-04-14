
train_data = DataGenerator(data, transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])   # Normalize for grayscale images
]))

