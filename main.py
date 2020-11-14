from sparktorch import serialize_torch_obj, SparkTorch
from pyspark.sql import SparkSession
from model import UNet
from loss import soft_dice_loss
import torch
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.pipeline import Pipeline

data_train_path = '/home/deepak/Courseworks/CS505_Big_Data/Project/mit_100x64x64.csv'

spark = SparkSession.builder.appName("UNet").master('local[2]').getOrCreate()

network = UNet(1)

torch_obj = serialize_torch_obj(
    model = network,
    criterion=soft_dice_loss,
    optimizer=torch.optim.Adam,
    lr=0.0001
)

spark_model = SparkTorch(
    inputCol='features',
    labelCol='labels',
    predictionCol='predictions',
    torchObj=torch_obj,
    iters=10,
    verbose=1
)

print("Ran successfully")

data_train = spark.read.option("inferSchema","true").option("maxColumns",64*64*4).csv(data_train_path)

features_size = 64*64*3
va1 = VectorAssembler(inputCols=data_train.columns[:features_size],
                      outputCol='features')
va2 = VectorAssembler(inputCols=data_train.columns[features_size:],
                      outputCol='labels')

p = Pipeline(stages=[va1, va2, spark_model]).fit(data_train)
p.save('unet')
