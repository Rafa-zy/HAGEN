import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.hagen_cell import DCGRUCell

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Seq2SeqAttrs:
    def __init__(self, **model_kwargs):
        self.max_diffusion_step = int(model_kwargs.get('max_diffusion_step', 2))
        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        self.filter_type = model_kwargs.get('filter_type', 'laplacian')
        self.num_nodes = int(model_kwargs.get('num_nodes', 1))
        self.num_rnn_layers = int(model_kwargs.get('num_rnn_layers', 1))
        self.rnn_units = int(model_kwargs.get('rnn_units'))
        self.hidden_state_size = self.num_nodes * self.rnn_units


class EncoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, **model_kwargs)
        self.input_dim = int(model_kwargs.get('input_dim', 1))
        self.seq_len = int(model_kwargs.get('seq_len'))
        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(self.rnn_units, self.max_diffusion_step, self.num_nodes,
                       filter_type=self.filter_type) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs, adj_mx, nodevec1, nodevec2, hidden_state=None):
        batch_size, _ = inputs.size()
        if hidden_state is None:
            hidden_state = torch.zeros((self.num_rnn_layers, batch_size, self.hidden_state_size), device=device)
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state, dw = dcgru_layer(output, hidden_state[layer_num], adj_mx)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state
        return output, torch.stack(hidden_states)


class DecoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, **model_kwargs)
        self.dim_poi = 15
        self.output_dim = int(model_kwargs.get('output_dim', 1))
        self.horizon = int(model_kwargs.get('horizon', 1))
        self.projection_layer = nn.Linear((self.rnn_units+self.dim_poi), self.output_dim)
        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(self.rnn_units, self.max_diffusion_step, self.num_nodes,
                       filter_type=self.filter_type) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs, adj_mx, nodevec1, nodevec2, POI_feat, labels, hidden_state=None):
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state, dw = dcgru_layer(output, hidden_state[layer_num], adj_mx)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state
        batch_size = output.size(0)
        output = output.reshape(batch_size*self.num_nodes, -1)
        dim_poi = POI_feat.size(1)
        poi_ex = POI_feat.expand(batch_size, self.num_nodes, dim_poi)
        poi_re = poi_ex.reshape(batch_size * self.num_nodes,-1)
        output_cat = torch.cat([output, poi_re], dim=1)
        projected = self.projection_layer(output_cat.view(-1, (self.rnn_units + dim_poi)))
        output = projected.view(-1, self.num_nodes * self.output_dim)
        return output, torch.stack(hidden_states)


class graph_constructor(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None, emb_dir='./embedding_chi.npy'):
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            emb_raw = np.load(emb_dir)
            embedding_la = torch.FloatTensor(emb_raw)
            self.emb1 = nn.Embedding.from_pretrained(embedding_la)
            self.emb2 = nn.Embedding.from_pretrained(embedding_la)
            self.lin1 = nn.Linear(dim,dim)
            self.lin2 = nn.Linear(dim,dim)
            self.lin3 = nn.Linear(self.nnodes, self.nnodes)
        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat
        self.rand_graph = torch.randn(113, 113)

    def forward(self, idx):
        self.emb1 = self.emb1.to(device)
        self.emb2 = self.emb2.to(device)
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))
        a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1,t1 = adj.topk(self.k,1)
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj*mask
        return adj, nodevec1, nodevec2

    def fullA(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))
        a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        return adj


class HAGENModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, logger, **model_kwargs):
        super().__init__()
        Seq2SeqAttrs.__init__(self, **model_kwargs)
        num_crime = model_kwargs.get('input_dim')
        subgraph_size = model_kwargs.get('subgraph_size')
        node_dim = model_kwargs.get('node_dim')
        tanhalpha = model_kwargs.get('tanhalpha')
        static_feat = None
        emb_dirr = model_kwargs.get('emb_dir')
        crime_emb_dirr = model_kwargs.get('crime_emb_dir')
        poi_dirr = model_kwargs.get('poi_dir')

        self.gc = graph_constructor(self.num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha, static_feat=static_feat, emb_dir=emb_dirr)
        self.idx = torch.arange(self.num_nodes).to(device)
        self.cidx = torch.arange(num_crime).to(device)
        self.gl = model_kwargs.get('graph_learning', True)
        self.model_kwargs = model_kwargs
        self.encoder_model = EncoderModel(**model_kwargs)
        self.decoder_model = DecoderModel(**model_kwargs)
        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        self.use_curriculum_learning = bool(model_kwargs.get('use_curriculum_learning', False))
        self._logger = logger

        crime_emb_raw = np.load(crime_emb_dirr)
        embedding_crime = torch.FloatTensor(crime_emb_raw) 
        self.emb_crime = nn.Embedding.from_pretrained(embedding_crime)
        self.projection = torch.nn.Linear(self.num_nodes, self.num_nodes)
        self.poi_feat = torch.FloatTensor(np.load(poi_dirr))

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def encoder(self, inputs, adj_mx, nodevec1, nodevec2):
        encoder_hidden_state = None
        for t in range(self.encoder_model.seq_len):
            _, encoder_hidden_state = self.encoder_model(inputs[t], adj_mx, nodevec1, nodevec2, encoder_hidden_state)
        return encoder_hidden_state

    def decoder(self, encoder_hidden_state, adj_mx, nodevec1, nodevec2, POI_feat, labels=None, batches_seen=None):
        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros((batch_size, self.num_nodes * self.decoder_model.output_dim), device=device)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol
        outputs = []
        for t in range(self.decoder_model.horizon):
            decoder_output, decoder_hidden_state = self.decoder_model(decoder_input, adj_mx, nodevec1, nodevec2, POI_feat, labels, decoder_hidden_state)
            decoder_input = decoder_output
            outputs.append(decoder_output)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = labels[t]
        outputs = torch.stack(outputs)
        return outputs

    def forward(self, inputs, labels=None, batches_seen=None):
        if self.gl:
            adj_mx, nodevec1, nodevec2 = self.gc(self.idx)
        crime_vector = self.emb_crime(self.cidx)
        nodevec1 = self.projection(nodevec1.permute(1,0)).permute(1,0) 
        nodevec2 = self.projection(nodevec2.permute(1,0)).permute(1,0)
        weight_mx1 = torch.mm(nodevec1, crime_vector.transpose(1,0)) 
        weight_mx2 = torch.mm(nodevec2, crime_vector.transpose(1,0))
        weight_mx1 = F.softmax(weight_mx1,dim =1)
        weight_mx2 = F.softmax(weight_mx2,dim =1)
        weight_mx = torch.add(weight_mx1, weight_mx2) / 2
        seq_len = int(inputs.size(0))
        batch_size = int(inputs.size(1))
        num_category = 8
        num_sensor = self.num_nodes
        weighted_inputs = torch.zeros([seq_len*batch_size,num_sensor,num_category])
        inputs = inputs.reshape(-1,num_category*num_sensor)
        inputs = inputs.reshape(-1,num_sensor,num_category)
        i = 0
        for mat in inputs:
            weighted_inputs[i,:,:] = torch.mul(weight_mx,mat)
            i += 1
        weighted_inputs = weighted_inputs.reshape(seq_len, batch_size, -1)
        weighted_inputs = F.sigmoid(weighted_inputs)

        encoder_hidden_state = self.encoder(weighted_inputs, adj_mx, nodevec1, nodevec2)
        self._logger.debug("Encoder complete, starting decoder")

        POI_feat = self.poi_feat
        outputs = self.decoder(encoder_hidden_state, adj_mx, nodevec1, nodevec2, POI_feat, labels, batches_seen=batches_seen)

        self._logger.debug("Decoder complete")
        if batches_seen == 0:
            self._logger.info("Total trainable parameters {}".format(count_parameters(self)))
        if(labels != None):
            return outputs, adj_mx
        else:
            return outputs
