import torch.nn as nn
import torch

class PositionalEncoder(nn.Module):
  def __init__(self, d_model , **kwargs) -> None:
    super().__init__( **kwargs)
    self.d_model = d_model
    # Growable cache. Starts empty and expands when a longer sequence is encountered.
    self.register_buffer('pos_enc_cache', torch.empty((0, d_model)))


  def get_positional_encoding(self,T,device='cpu',dtype=torch.float32):
    """
    T: sequence length
    d_model: model dimensions, also same as embedding dims
    """
    # Alot of things from this function can be precomputed ,stored and reused for better performance.
    i = torch.arange(0,self.d_model,2, device=device, dtype=dtype) # [0,2,4,...d_model//2] - only even indices since we are doing sin and cos in pairs
    # this is written as ^-1 so it can be multiplied insted of div
    div = torch.exp(-torch.log(torch.tensor(10000,device=device, dtype=dtype))*i/self.d_model).unsqueeze(0) # (1, d_model//2)
    pos = torch.arange(0,T,1,device=device, dtype=dtype).unsqueeze(1) # [T,1]
    # print(f"pos shape {pos.shape} , div shape {div.shape}")
    angles = pos * div
    positional_encodings = torch.zeros((T,self.d_model), device=device, dtype=dtype)
    positional_encodings[:,0::2] = torch.sin(angles)
    positional_encodings[:,1::2] = torch.cos(angles)
    return positional_encodings

  def _ensure_cache(self, seq_len, device, dtype):
    cache_len = self.pos_enc_cache.shape[0]
    same_device = self.pos_enc_cache.device == device
    same_dtype = self.pos_enc_cache.dtype == dtype

    if (not same_device) or (not same_dtype):
      # Rebuild cache on current execution device/dtype.
      self.pos_enc_cache = self.get_positional_encoding(seq_len, device=device, dtype=dtype)
      return

    if seq_len > cache_len:
      # Grow from current longest length to new length.
      start = cache_len
      growth = seq_len - cache_len

      i = torch.arange(0,self.d_model,2, device=device, dtype=dtype)
      div = torch.exp(-torch.log(torch.tensor(10000,device=device, dtype=dtype))*i/self.d_model).unsqueeze(0)
      pos = torch.arange(start,start + growth,1,device=device, dtype=dtype).unsqueeze(1)
      angles = pos * div

      extra = torch.zeros((growth,self.d_model), device=device, dtype=dtype)
      extra[:,0::2] = torch.sin(angles)
      extra[:,1::2] = torch.cos(angles)

      self.pos_enc_cache = torch.cat((self.pos_enc_cache, extra), dim=0)

  def forward(self,X):
    input_seq_len = X.shape[1]
    self._ensure_cache(input_seq_len, device=X.device, dtype=X.dtype)
    pos_enc = self.pos_enc_cache[:input_seq_len,:]
    return X + pos_enc.unsqueeze(0)