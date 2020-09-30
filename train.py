import os
from nets import blaze_palm
from utils import loss_function
from utils.data_generator import DataGenerator

# Training Config
BATCH_SIZE = 32
EPOCHS = 1000
TRAIN_IMAGE_DIR = os.path.join('.', 'dataset', 'image')
TRAIN_ANNOTATION_DIR = os.path.join('.', 'dataset', 'annotation')
VAL_IMAGE_DIR = os.path.join('.', 'dataset', 'image')
VAL_ANNOTATION_DIR = os.path.join('.', 'dataset', 'annotation')

# Create Model
model = blaze_palm.build_blaze_palm_model()
ssd_loss = loss_function.SSDLoss(alpha=1. / 256.)
model.compile(optimizer='adam', loss=ssd_loss.compute_loss)

# Prepare Data Generator
train_data_generator = DataGenerator(image_dir=TRAIN_IMAGE_DIR,
                                     annotation_dir=TRAIN_ANNOTATION_DIR,
                                     batch_size=BATCH_SIZE)
val_data_generator = DataGenerator(image_dir=TRAIN_IMAGE_DIR,
                                   annotation_dir=TRAIN_ANNOTATION_DIR,
                                   batch_size=BATCH_SIZE)

# Fit Model
history = model.fit(x=train_data_generator, epochs=1000)

# Save Model
model.save(os.path.join('model', 'blaze_palm_model.h5'))
