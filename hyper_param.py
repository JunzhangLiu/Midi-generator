MAX_NOTE,MIN_NOTE = 109,21

NOTE_PRECISION = 64
LATENT_DIM = 128
INPUT_DIM = MAX_NOTE-MIN_NOTE

SECTION = 8 
# TIME_STEP = SECTION * TICKS_PER_MEASURE
PIX_PER_SECTION = 192 #192 is least common multiple of 64 and 96, which are multiples of most time signatures
TIME_STEP = PIX_PER_SECTION*SECTION
# TIME_STEP = 512
DROP_OUT_RATE = 0.1
BATCH_SIZE = 256

TRAIN_STEPS = 70
EPOCHS = 32
COMPONENTS_MEAN_SAVE_LOCATION = "./latent_distribution/components_mean.npy"
COMPONENTS_STD_SAVE_LOCATION = "./latent_distribution/components_std.npy"
COMPONENTS_SAVE_LOCATION = "./latent_distribution/components.npy"
DATA_MEAN_SAVE_LOCATION = "./latent_distribution/components_mean.npy"
DATA_STD_SAVE_LOCATION = "./latent_distribution/components_std.npy"
EPS_STD = 0.02
KL_LOSS_WEIGHT= 0.1