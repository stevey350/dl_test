
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.reducers import ThresholdReducer
from pytorch_metric_learning.regularizers import LpRegularizer
from pytorch_metric_learning import losses

loss_func = losses.TripletMarginLoss(distance = CosineSimilarity(),
             reducer = ThresholdReducer(high=0.3),
              embedding_regularizer = LpRegularizer())

