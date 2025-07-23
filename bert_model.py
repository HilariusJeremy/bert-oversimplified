import torch
import torch.nn as nn
import math

config = {
    'vocab_size': 30522, 
    'hidden_size': 768,                 # embedding dimension for each token index 
    'max_position_embeddings':  512,    # max. sequence length
    'pad_token_id': 0,                  # index of padding token in the vocabulary  
    'type_vocab_size': 2,
    'hidden_dropout_prob': 0.1,
    'layer_norm_eps': 1e-12,
    'num_attention_heads': 12,
    'attention_probs_dropout_prob': 0.1,
    'num_hidden_layers': 12,

}

class CustomBertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(num_embeddings=config['vocab_size'], 
                                            embedding_dim=config['hidden_size'], 
                                            padding_idx=config['pad_token_id'])
        
        self.token_type_embeddings = nn.Embedding(num_embeddings=config['type_vocab_size'],
                                                  embedding_dim=config['hidden_size'])
        
        self.position_embeddings = nn.Embedding(num_embeddings=config['max_position_embeddings'],
                                                embedding_dim=config['hidden_size'])
        
        self.register_buffer(name='position_ids', tensor=torch.arange(end=config['max_position_embeddings']).expand(1, -1), persistent=False)
        self.dropout = nn.Dropout(p=config['hidden_dropout_prob'])
        self.LayerNorm = nn.LayerNorm(normalized_shape=config['hidden_size'], eps=config['layer_norm_eps'])

    # Assume input_ids and token_type_ids have shape of [batch_size, max_seq_len_in_batch]
    # Here, we'll let max_seq_len to be the maximum length of the sequence in the batch
    def forward(self, input_ids, token_type_ids):
        input_shape = input_ids.shape
        max_seq_len = input_shape[1]
        position_ids = self.position_ids[:, :max_seq_len]
        input_embeds = self.word_embeddings(input_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)
        position_embeds = self.position_embeddings(position_ids)
        embeddings = input_embeds + token_type_embeds + position_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class CustomBertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config['num_attention_heads']
        assert (config['hidden_size'] % self.num_attention_heads == 0)
        self.head_hidden_size = config['hidden_size'] // self.num_attention_heads
        self.all_head_size = self.head_hidden_size * self.num_attention_heads
        self.dropout = nn.Dropout(p=config['attention_probs_dropout_prob'])

        # Initialize key, query, value
        self.key = nn.Linear(config['hidden_size'], self.all_head_size)
        self.query = nn.Linear(config['hidden_size'], self.all_head_size)
        self.value = nn.Linear(config['hidden_size'], self.all_head_size)

    # Assume hidden_state with dimension [batch_size, max_seq_len, hidden_size]
    # Assume attention_mask with dimension [batch_size, max_seq_len]
    def forward(self, hidden_state, attention_mask):
        # Get the input shape
        batch_size, max_seq_len, hidden_size = hidden_state.shape

        # Reshape key, query, and value to [batch_size, seq_len, attn_head, head_hidden_size] -> [batch_size, attn_head, seq_len, head_hidden_size]
        q = self.query(hidden_state).view((batch_size, -1, self.num_attention_heads, self.head_hidden_size)).permute((0, 2, 1, 3))
        k = self.key(hidden_state).view((batch_size, -1, self.num_attention_heads, self.head_hidden_size)).permute((0, 2, 1, 3))
        v = self.value(hidden_state).view((batch_size, -1, self.num_attention_heads, self.head_hidden_size)).permute((0, 2, 1, 3))    
        # Multiply query and key transpose -> [batch_size, attn_head, seq_len, head_hidden_size] x [batch_size, attn_head, head_hidden_size, seq_len]
        # Final shape = [batch_size, attn_head, seq_len, seq_len]
        attention_score = torch.matmul(q, k.transpose(-2, -1))
        # Divide by sqrt of head_hidden_size
        attention_score = attention_score / math.sqrt(self.head_hidden_size)        
        # Reshape attention_mask to shape [batch_size, 1, 1, seq_len]
        attention_mask = attention_mask[:, None, None, :]
        attention_mask = (1.0 - attention_mask) * -10000.0
        attention_score += attention_mask
        # Apply softmax row wise 
        attention_prob = attention_score.softmax(dim=-1) 
        # Apply dropout
        attention_prob = self.dropout(attention_prob)
        # Multiply with v
        context_layer = torch.matmul(attention_prob, v)
        # [batch_size, attn_head, seq_len, head_hidden_size] 
        context_layer = context_layer.transpose(1, 2).contiguous()
        context_layer = context_layer.view((batch_size, -1, self.all_head_size))
        
        return context_layer, attention_prob
    
class CustomBertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.dropout = nn.Dropout(p=config['hidden_dropout_prob'])
        self.LayerNorm = nn.LayerNorm(config['hidden_size'], eps=config['layer_norm_eps'])
        
    def forward(self, hidden_state, input_tensor):
        # Apply dense to hidden_state
        hidden_state = self.dense(hidden_state)
        # Apply dropout to hidden_state
        hidden_state = self.dropout(hidden_state)
         # Apply LayerNorm to hidden_state + input_tensor
        hidden_state = self.LayerNorm(hidden_state + input_tensor)
        return hidden_state

class CustomBertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = CustomBertSelfAttention(config)
        self.output = CustomBertSelfOutput(config)
    
    def forward(self, hidden_states, attention_mask):
        self_output = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_output[0], hidden_states)
        return (attention_output, self_output[1])

class CustomBertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config['intermediate_size'], config['hidden_size'])
        self.dropout = nn.Dropout(config['hidden_dropout_prob'])
        self.LayerNorm = nn.LayerNorm(config['hidden_size'], eps=config['layer_norm_eps'])
    
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return self.LayerNorm(hidden_states + input_tensor)
     
class CustomBertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config['hidden_size'], config['intermediate_size'])
        self.act_fn = nn.GELU()
                   
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        return hidden_states

class CustomBertLayer(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.attention = CustomBertAttention(config)
		self.intermediate = CustomBertIntermediate(config)	
		self.output = CustomBertOutput(config)

	def forward(self, hidden_state, attention_mask):
		attention_output = self.attention(hidden_state, attention_mask)
		context_layer, attention_prob = attention_output
		intermediate_output = self.intermediate(context_layer)
		output = self.output(intermediate_output, context_layer)
		return output, attention_prob

class CustomBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([CustomBertLayer(config) for i in range(config['num_hidden_layers'])])

    def forward(self, hidden_state, attention_mask):
        for i, layer in enumerate(self.layers): 
            hidden_state, _ = layer(hidden_state, attention_mask)
        return hidden_state    

class CustomBertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.act_fn = nn.Tanh()

    def forward(self, hidden_state):
        cls_hidden_state = hidden_state[:, 0, :]
        cls_hidden_state = self.dense(cls_hidden_state)
        cls_hidden_state = self.act_fn(cls_hidden_state)
        return cls_hidden_state


class CustomBertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = CustomBertEmbeddings(config)
        self.encoder = CustomBertEncoder(config)
        self.pooler = CustomBertPooler(config)
       
    def forward(self, input_ids, token_type_ids, attention_mask):
        embeddings = self.embeddings(input_ids, token_type_ids)
        hidden_state = self.encoder(embeddings, attention_mask)
        cls_pooler = self.pooler(hidden_state)
        return hidden_state, cls_pooler



        
    