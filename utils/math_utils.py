import numpy as np
import torch

class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

class MinMaxScaler:

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data):
        return (data * (self.max - self.min)) + self.min

def MAE_torch(pred, true, mask_value=0.):
    if np.isnan(mask_value):
        mask = ~torch.isnan(true)
    else:
        mask = (true!=mask_value)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(pred-true)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def MSE_torch(pred, true, mask_value=0.):
    if np.isnan(mask_value):
        mask = ~torch.isnan(true)
    else:
        mask = (true!=mask_value)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (pred-true)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def RMSE_torch(pred, true, mask_value=0.):
    return torch.sqrt(MSE_torch(preds=pred, true=true, mask_value=mask_value))

def MAPE_torch(pred, true, mask_value=0.):
    if np.isnan(mask_value):
        mask = ~torch.isnan(true)
    else:
        mask = (true!=mask_value)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs((pred-true)/true)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def mask_np(array, mask_value):
    if np.isnan(mask_value):
        return (~np.isnan(array)).astype('float32')
    else:
        return np.not_equal(array, mask_value).astype('float32')

def MAE_np(pred, true, mask_value=0.):
    if np.isnan(mask_value):
        mask = ~np.isnan(true)
    else:
        mask = (true!=mask_value)
    mask = mask.astype(np.float32)
    mask /=  np.mean((mask))
    mask = np.where(np.isnan(mask), np.zeros_like(mask), mask)
    loss = np.abs(pred-true)
    loss = loss * mask
    loss = np.where(np.isnan(loss), np.zeros_like(loss), loss)
    return np.mean(loss)

def MSE_np(pred, true, mask_value=0.):
    if np.isnan(mask_value):
        mask = ~np.isnan(true)
    else:
        mask = (true!=mask_value)
    mask = mask.astype(np.float32)
    mask /= np.mean((mask))
    mask = np.where(np.isnan(mask), np.zeros_like(mask), mask)
    loss = (pred-true)**2
    loss = loss * mask
    loss = np.where(np.isnan(loss), np.zeros_like(loss), loss)
    return np.mean(loss)

def RMSE_np(pred, true, mask_value=0.):
    return np.sqrt(MSE_np(pred=pred, true=true, mask_value=mask_value))

def MAPE_np( pred, true, mask_value=0.):
    if np.isnan(mask_value):
        mask = ~np.isnan(true)
    else:
        mask = (true!=mask_value)
    mask = mask.astype(np.float32)
    mask /=  np.mean((mask))
    mask = np.where(np.isnan(mask), np.zeros_like(mask), mask)
    loss = np.abs((pred-true)/true)
    loss = loss * mask
    loss = np.where(np.isnan(loss), np.zeros_like(loss), loss)
    loss = np.where(loss>100, np.zeros_like(loss), loss)
    return np.mean(loss)

def evaluate(pred, true, mask1=0., mask2=0.):
    #mask1 filter the very small value, mask2 filter the value lower than a defined threshold
    assert type(pred) == type(true)
    if type(pred) == np.ndarray:
        mae  = MAE_np(pred, true, mask1)
        rmse = RMSE_np(pred, true, mask1)
        mape = MAPE_np(pred, true, mask2)

    elif type(pred) == torch.Tensor:
        mae  = MAE_torch(pred, true, mask1)
        rmse = RMSE_torch(pred, true, mask1)
        mape = MAPE_torch(pred, true, mask2)

    else:
        raise TypeError
    return mape, mae, rmse
    