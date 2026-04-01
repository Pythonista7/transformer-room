# A rough param count approx of a vanilla transformer
d_model = 768
n_layers = 12
vocab_size=50_258
model_tying = True


# Input Embedding Proj
input_proj = vocab_size * d_model


# Attention params
attn_in_proj = d_model * d_model * 3
attn_out_proj = d_model * d_model

# decoder output projs
dec_out_proj1 = d_model * d_model * 4
dec_out_proj2 = d_model * 4 * d_model

# decoder block params
dec_block_params = attn_in_proj + attn_out_proj + dec_out_proj1 + dec_out_proj2

layers_param_count = n_layers * dec_block_params

# final output projs
final_out_proj = d_model * vocab_size

total_param_count = input_proj + layers_param_count + final_out_proj if not model_tying else input_proj + layers_param_count


print(f"""
      Embedding Projs { ' & reused as final output proj' if model_tying else ''} : {input_proj}
      ================================================
      Attention Params: {attn_in_proj + attn_out_proj}
      Decoder Block Params: {dec_block_params}
      Decoder Output Projs: {dec_out_proj1 + dec_out_proj2}
      ================================================
      
      Dec Layers: {n_layers}
      Total Dec Layers Params: {layers_param_count}
      
      ================================================
      Final Output Projs: {final_out_proj}
      Total Params: {total_param_count}
      """,)