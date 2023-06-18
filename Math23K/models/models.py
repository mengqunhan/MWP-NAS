from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel,BertConfig
import torch
import torch.nn as nn
import numpy as np
import copy


class PositionEmbeddings(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.max_position_embeddings = args.max_pos_num
        self.position_embeddings = nn.Embedding(self.max_position_embeddings, args.hidden_size)
        self.create_sinusoidal_embeddings(
            n_pos=self.max_position_embeddings, dim=args.hidden_size, out=self.position_embeddings.weight
        )
        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.pos_dropout)
        self.register_buffer(
            "position_ids", torch.arange(self.max_position_embeddings).expand((1, 1, -1)), persistent=False
        )

    def create_sinusoidal_embeddings(self, n_pos: int, dim: int, out: torch.Tensor):
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)])
        out.requires_grad = False
        out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()

    def forward(self, sub_goal: torch.Tensor, seq_length) -> torch.Tensor:
        """
        Parameters:
            input_ids: torch.tensor(bs, max_seq_length) The token ids to embed.
        Returns: torch.tensor(bs, max_seq_length, dim) The embedded tokens (plus position embeddings, no token_type
        embeddings)
        """
        # seq_length = input_ids.size(1)
        batch_size, num_node, _ = sub_goal.size()

        # Setting the position-ids to the registered buffer in constructor, it helps
        # when tracing the model without passing position-ids, solves
        # isues similar to issue #5664
        if hasattr(self, "position_ids"):
            position_ids = self.position_ids[:, :, :seq_length]  # (1,1,seq_len)
        else:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=sub_goal.device)  # (max_seq_length)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)  # (bs, max_seq_length)

        # word_embeddings = self.word_embeddings(input_ids)  # (bs, max_seq_length, dim)
        word_embeddings = sub_goal.unsqueeze(2).repeat(1, 1, seq_length, 1)  # (batch_size,num_node,seq_len,hidden_size)
        position_embeddings = self.position_embeddings(position_ids)  # (1,1,seq_len,hidden_size)

        embeddings = word_embeddings + position_embeddings  # (bs, num_node, seq_len, dim)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class PointerNetwork(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embedding_dim = args.hidden_size

        self.lstm_cell = nn.LSTMCell(
            input_size=2 * self.embedding_dim, hidden_size=self.embedding_dim
        )
        self.dropout = nn.Dropout(args.ptr_dropout)

        self.reference = nn.Linear(self.embedding_dim, 1)
        self.decoder_weights = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.encoder_weights = nn.Linear(self.embedding_dim, self.embedding_dim)

    def forward(self, candidate: torch.tensor, inter_output: torch.tensor,
                candidate_mask: torch.tensor) -> torch.tensor:
        batch_size, n_token, hidden_dim = candidate.size()
        num_pos = inter_output.size(1)

        decoder_query = self.decoder_weights(inter_output)  # (batch_size,num_pos,hidden_dim)
        token_embeddings = self.encoder_weights(candidate)  # (batch_size,num_candidate,hidden_dim)

        decoder_query = decoder_query.unsqueeze(2).repeat(1, 1, n_token,
                                                          1)  # (batch_size,num_pos,num_candidate,hidden_dim)
        token_embeddings = token_embeddings.unsqueeze(1).repeat(1, num_pos, 1,
                                                                1)  # (batch_size,num_pos,num_candidate,hidden_dim)
        comparison = torch.tanh(decoder_query + token_embeddings)  # (batch_size,num_pos,num_candidate,hidden_dim)
        candidate_mask = candidate_mask.unsqueeze(1).repeat(1, num_pos, 1)  # (batch_size,num_pos,num_candidate)
        logits = self.reference(comparison).reshape(batch_size, num_pos, n_token).masked_fill(candidate_mask,
                                                                                              -1e9)  # (batch_size,num_pos,num_candidate)
        return logits, comparison


class UniversalModel(BertPreTrainedModel):
    def __init__(self, config: BertConfig,
                 args
                 ):
        super().__init__(config)
        self.config = config
        self.bert = BertModel(config)
        #self.operators = nn.Parameter(torch.randn(args.num_start, args.hidden_size))
        self.num_constant=args.num_constant
        self.num_start = args.num_start
        if self.num_constant > 0:
            self.const_rep = nn.Parameter(torch.randn(self.num_constant, args.hidden_size))

        self.non_number_rep = nn.Parameter(torch.randn(args.num_start, args.hidden_size))

        self.cls_ffn = nn.Linear(args.hidden_size, args.hidden_size)
        self.sub_goal_ffn = nn.Linear(args.hidden_size, args.hidden_size)
        self.candidate_ffn = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.ReLU(),
            nn.LayerNorm(args.hidden_size, eps=args.layer_norm_eps),
            nn.Dropout(args.hidden_dropout)
        )

        self.self_attn = nn.MultiheadAttention(args.hidden_size, args.num_attn_heads, batch_first=True)
        self.inter_attn = nn.MultiheadAttention(args.hidden_size, args.num_attn_heads, batch_first=True)
        self.residusal_LN = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_eps)
        self.inter_attn1 = nn.MultiheadAttention(args.hidden_size, args.num_attn_heads, batch_first=True)
        self.residusal_LN1 = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_eps)
        self.position_embedding = PositionEmbeddings(args)
        #self.ptr_net = PointerNetwork(args, args.hidden_size)
        self.ptr_net = PointerNetwork(args)

        self.format_ffn = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.LeakyReLU(args.leaky_relu_slope),
            nn.LayerNorm(args.hidden_size, eps=args.layer_norm_eps),
            nn.Dropout(args.hidden_dropout),
            nn.Linear(args.hidden_size, args.hidden_size // 2),
            nn.LeakyReLU(args.leaky_relu_slope),
            nn.Linear(args.hidden_size // 2, args.hidden_size // 4),
            nn.LeakyReLU(args.leaky_relu_slope),
            nn.Linear(args.hidden_size // 4, args.num_format)
        )

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            variable_indexs_start=None,
            variable_indexs_end=None,
            num_variables=None,
            variable_index_mask=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            max_height=-1,
            max_width=-1,
            mtree_node_data=None,
            mtree_node_format=None,
            word2index=None,
            criterion=None
            ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        pooler_output = outputs.pooler_output
        batch_size, sent_len, hidden_size = outputs.last_hidden_state.size()
        _, max_num_variable = variable_indexs_start.size()
        var_sum = (variable_indexs_start - variable_indexs_end).sum()
        var_start_hidden_states = torch.gather(outputs.last_hidden_state, 1,
                                               variable_indexs_start.unsqueeze(-1).expand(batch_size, max_num_variable,
                                                                                          hidden_size))
        if var_sum != 0:
            var_end_hidden_states = torch.gather(outputs.last_hidden_state, 1,
                                                 variable_indexs_end.unsqueeze(-1).expand(batch_size, max_num_variable,
                                                                                          hidden_size))
            var_hidden_states = var_start_hidden_states + var_end_hidden_states
        else:
            var_hidden_states = var_start_hidden_states

        non_number_hidden_states = self.non_number_rep.unsqueeze(0).expand(batch_size, self.num_start, hidden_size)
        constant_hidden_states = self.const_rep.unsqueeze(0).expand(batch_size, self.num_constant, hidden_size)
        candidate_hidden_states = torch.cat([non_number_hidden_states, constant_hidden_states, var_hidden_states],
                                            dim=1)
        num_candidate = num_variables + self.num_constant + self.num_start
        max_num_candidate = max_num_variable + self.num_constant + self.num_start
        non_number_idx_mask = torch.ones((batch_size, self.num_constant + self.num_start),
                                         device=variable_indexs_start.device)
        candidate_index_mask = torch.cat([non_number_idx_mask, variable_index_mask], dim=1)
        candidate_attn_mask = candidate_index_mask == 0
        #return pooler_output,candidate_hidden_states,candidate_attn_mask

        goals = self.cls_ffn(pooler_output).unsqueeze(1)
        cur_height = 1

        batch_item_loss = [0 for _ in range(batch_size)]
        batch_item_format_loss = [0 for _ in range(batch_size)]
        batch_item_num = [0 for _ in range(batch_size)]
        batch_item_format_num = [0 for _ in range(batch_size)]

        while cur_height<max_height:
            cur_node_start=(max_width**(cur_height-1)-1)//(max_width-1)
            cur_node_end=(max_width**(cur_height)-1)//(max_width-1)
            cur_children_end=(max_width**(cur_height+1)-1)//(max_width-1)
            cur_goal=goals[:,:cur_node_end-cur_node_start,:]
            cur_children=mtree_node_data[:,cur_node_end:cur_children_end]
            cur_children_format=mtree_node_format[:,cur_node_end:cur_children_end]

            #self attention
            pos_embedding=self.position_embedding(cur_goal,max_width)
            pos_embedding_for_attn=pos_embedding.view(pos_embedding.size(0),-1,pos_embedding.size(-1))
            self_attn_output,_=self.self_attn(pos_embedding_for_attn,pos_embedding_for_attn,pos_embedding_for_attn)

            candidate_hidden_states_ffn=self.candidate_ffn(candidate_hidden_states)
            # inter_attn_output,_=self.inter_attn(query=self_attn_output,
            #                                     key=candidate_hidden_states_ffn,
            #                                     value=candidate_hidden_states_ffn,
            #                                     key_padding_mask=candidate_attn_mask)
            inter_attn_output, _ = self.inter_attn(query=self_attn_output,
                                                   key=candidate_hidden_states_ffn,
                                                   value=candidate_hidden_states_ffn,
                                                   key_padding_mask=candidate_attn_mask)
            inter_attn_output_res = self.residusal_LN(inter_attn_output + self_attn_output)
            inter_attn_output, _ = self.inter_attn1(query=inter_attn_output_res,
                                                    key=candidate_hidden_states_ffn,
                                                    value=candidate_hidden_states_ffn,
                                                    key_padding_mask=candidate_attn_mask)
            inter_attn_output = self.residusal_LN1(inter_attn_output + inter_attn_output_res)


            probilities,comparision=self.ptr_net(candidate_hidden_states,inter_attn_output,candidate_attn_mask)
            # probilities=probilities.view(-1,probilities.size(-1))

            #target_mask=cur_children[:,0]==word2index['PAD']
            target_mask = cur_children == word2index['PAD']
            #target_mask=target_mask.unsqueeze(-1).expand(cur_children.size())
            target=copy.deepcopy(cur_children)
            target=target.masked_fill(target_mask,-1)
            # target=target.view(-1)

            for i in range(batch_size):
                candidate_loss=criterion(probilities[i],target[i])
                batch_item_loss[i]+=(candidate_loss if (target_mask[i]==False).sum().item()!=0 else 0)
                batch_item_num[i]+=(target_mask[i]==False).sum().item()

            # candidate_loss=criterion(probilities,target)
            # batch_item_num+=(target_mask==False).sum().item()
            # candidate_loss=candidate_loss if (target_mask==False).sum().item()!=0 else 0
            # batch_item_loss+=candidate_loss

            classifier_input=inter_attn_output
            classifier_logits=self.format_ffn(classifier_input)

            classifier_mask=cur_children<self.num_start
            classifier_target=cur_children_format.masked_fill(classifier_mask,-1)

            # classifier_logits_view=classifier_logits.view(-1,classifier_logits.size(-1))
            # classifier_target_view=classifier_target.view(-1)

            # classifier_loss=criterion(classifier_logits_view,classifier_target_view)
            # batch_item_format_num+=(classifier_mask==False).sum().item()
            # classifier_loss=classifier_loss if (classifier_mask==False).sum().item()!=0 else 0
            # batch_item_format_loss+=classifier_loss

            for i in range(batch_size):
                classifier_loss=criterion(classifier_logits[i],classifier_target[i])
                batch_item_format_loss[i]+=(classifier_loss if (classifier_mask[i]==False).sum().item()!=0 else 0)
                batch_item_format_num[i]+=(classifier_mask[i]==False).sum().item()

            assert inter_attn_output.size(0)==pos_embedding.size(0)
            # assert inter_attn_output.size(1)==pos_embedding.size(1)
            goals=self.sub_goal_ffn(inter_attn_output)
            cur_height+=1

        loss1=0
        loss2=0
        for i in range(batch_size):
            loss1+=batch_item_loss[i]/batch_item_num[i] if batch_item_num[i]!=0 else 0
            loss2+=batch_item_format_loss[i]/batch_item_format_num[i] if batch_item_format_num[i]!=0 else 0
        loss1=loss1/batch_size
        loss2=loss2/batch_size
        loss=loss1+loss2
        return loss,loss1,loss2


    def eval_forward(self,
                     input_ids=None,
                     attention_mask=None,
                     token_type_ids=None,
                     variable_indexs_start=None,
                     variable_indexs_end=None,
                     num_variables=None,
                     variable_index_mask=None,
                     position_ids=None,
                     head_mask=None,
                     inputs_embeds=None,
                     output_attentions=None,
                     output_hidden_states=None,
                     return_dict=None,
                     root_node_data=None,
                     root_node_format=None,
                     max_height=None,
                     max_width=None,
                     word2index=None,
                     target_mtree=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        pooler_output = outputs.pooler_output
        batch_size, sent_len, hidden_size = outputs.last_hidden_state.size()
        _, max_num_variable = variable_indexs_start.size()
        var_sum = (variable_indexs_start - variable_indexs_end).sum()
        var_start_hidden_states = torch.gather(outputs.last_hidden_state, 1,
                                               variable_indexs_start.unsqueeze(-1).expand(batch_size, max_num_variable,
                                                                                          hidden_size))
        if var_sum != 0:
            var_end_hidden_states = torch.gather(outputs.last_hidden_state, 1,
                                                 variable_indexs_end.unsqueeze(-1).expand(batch_size, max_num_variable,
                                                                                          hidden_size))
            var_hidden_states = var_start_hidden_states + var_end_hidden_states
        else:
            var_hidden_states = var_start_hidden_states

        non_number_hidden_states = self.non_number_rep.unsqueeze(0).expand(batch_size, self.num_start, hidden_size)
        constant_hidden_states = self.const_rep.unsqueeze(0).expand(batch_size, self.num_constant, hidden_size)
        candidate_hidden_states = torch.cat([non_number_hidden_states, constant_hidden_states, var_hidden_states],
                                            dim=1)
        num_candidate = num_variables + self.num_constant + self.num_start
        max_num_candidate = max_num_variable + self.num_constant + self.num_start
        non_number_idx_mask = torch.ones((batch_size, self.num_constant + self.num_start),
                                         device=variable_indexs_start.device)
        candidate_index_mask = torch.cat([non_number_idx_mask, variable_index_mask], dim=1)
        candidate_attn_mask = candidate_index_mask == 0

        goals = self.cls_ffn(pooler_output).unsqueeze(1)
        cur_height = 1
        predict = root_node_data
        predict_format = root_node_format
        while cur_height < max_height:
            cur_node_start = (max_width ** (cur_height - 1) - 1) // (max_width - 1)
            cur_node_end = (max_width ** (cur_height) - 1) // (max_width - 1)
            cur_children_end = (max_width ** (cur_height + 1) - 1) // (max_width - 1)
            cur_goal = goals[:, :cur_node_end - cur_node_start, :]

            pos_embedding = self.position_embedding(cur_goal, max_width)
            pos_embedding_for_attn = pos_embedding.view(pos_embedding.size(0), -1, pos_embedding.size(-1))
            self_attn_output, _ = self.self_attn(pos_embedding_for_attn, pos_embedding_for_attn, pos_embedding_for_attn)

            candidate_hidden_states_ffn = self.candidate_ffn(candidate_hidden_states)
            # inter_attn_output, _ = self.inter_attn(query=self_attn_output,
            #                                        key=candidate_hidden_states_ffn,
            #                                        value=candidate_hidden_states_ffn,
            #                                        key_padding_mask=candidate_attn_mask)
            inter_attn_output, _ = self.inter_attn(query=self_attn_output,
                                                   key=candidate_hidden_states_ffn,
                                                   value=candidate_hidden_states_ffn,
                                                   key_padding_mask=candidate_attn_mask)
            inter_attn_output_res = self.residusal_LN(inter_attn_output + self_attn_output)
            inter_attn_output, _ = self.inter_attn1(query=inter_attn_output_res,
                                                    key=candidate_hidden_states_ffn,
                                                    value=candidate_hidden_states_ffn,
                                                    key_padding_mask=candidate_attn_mask)
            inter_attn_output = self.residusal_LN1(inter_attn_output + inter_attn_output_res)

            probilities, comparision = self.ptr_net(candidate_hidden_states, inter_attn_output, candidate_attn_mask)
            pred_tokens = probilities.argmax(dim=-1).view(batch_size, -1)

            predict = torch.cat([predict, pred_tokens], dim=1)
            classifier_input = inter_attn_output
            classifier_logits = self.format_ffn(classifier_input)
            classifier_pred = classifier_logits.argmax(dim=-1)

            classifier_mask = pred_tokens < self.num_start
            classifier_predict = classifier_pred.masked_fill(classifier_mask, 0)
            predict_format = torch.cat([predict_format, classifier_predict], dim=1)

            assert inter_attn_output.size(0) == pos_embedding.size(0)
            # assert inter_attn_output.size(1)==pos_embedding.size(1)
            goals = self.sub_goal_ffn(inter_attn_output)
            cur_height += 1

        pad_idx = word2index['PAD']
        end_idx = word2index['EOS']
        self.pad_idx = pad_idx
        self.end_idx = end_idx
        predict_mtree = self.recover_mtree(predict, predict_format, max_height, max_width, 1, 0)
        node_right, format_right, mtree_right, result_batch = self.mtree_equal_batch(predict_mtree, target_mtree)
        return node_right, format_right, mtree_right, result_batch

    def evalue(self,
               input_ids=None,
               attention_mask=None,
               token_type_ids=None,
               variable_indexs_start=None,
               variable_indexs_end=None,
               num_variables=None,
               variable_index_mask=None,
               position_ids=None,
               head_mask=None,
               inputs_embeds=None,
               output_attentions=None,
               output_hidden_states=None,
               return_dict=None,
               root_node_data=None,
               root_node_format=None,
               max_height=None,
               max_width=None,
               word2index=None,
               target_mtree=None):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        pooler_output = outputs.pooler_output
        # mean pooling
        # token_embeddings = outputs.last_hidden_state[:, 1:, :]
        # input_mask_expanded = (attention_mask[:, 1:]).unsqueeze(-1).expand(token_embeddings.size()).float()
        # sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        # sum_mask = input_mask_expanded.sum(1)
        # sum_mask = torch.clamp(sum_mask, min=1e-9)
        # pooler_output = sum_embeddings / sum_mask

        batch_size, sent_len, hidden_size = outputs.last_hidden_state.size()
        _, max_num_variable = variable_indexs_start.size()
        var_sum = (variable_indexs_start - variable_indexs_end).sum()
        var_start_hidden_states = torch.gather(outputs.last_hidden_state, 1,
                                               variable_indexs_start.unsqueeze(-1).expand(batch_size, max_num_variable,
                                                                                          hidden_size))
        if var_sum != 0:
            var_end_hidden_states = torch.gather(outputs.last_hidden_state, 1,
                                                 variable_indexs_end.unsqueeze(-1).expand(batch_size, max_num_variable,
                                                                                          hidden_size))
            var_hidden_states = var_start_hidden_states + var_end_hidden_states
        else:
            var_hidden_states = var_start_hidden_states

        non_number_hidden_states = self.non_number_rep.unsqueeze(0).expand(batch_size, self.num_start, hidden_size)
        constant_hidden_states = self.const_rep.unsqueeze(0).expand(batch_size, self.num_constant, hidden_size)
        candidate_hidden_states = torch.cat([non_number_hidden_states, constant_hidden_states, var_hidden_states],
                                            dim=1)
        num_candidate = num_variables + self.num_constant + self.num_start
        max_num_candidate = max_num_variable + self.num_constant + self.num_start
        non_number_idx_mask = torch.ones((batch_size, self.num_constant + self.num_start),
                                         device=variable_indexs_start.device)
        candidate_index_mask = torch.cat([non_number_idx_mask, variable_index_mask], dim=1)
        candidate_attn_mask = candidate_index_mask == 0

        goals = self.cls_ffn(pooler_output).unsqueeze(1)
        cur_height = 1
        predict = root_node_data
        predict_format = root_node_format
        while cur_height < max_height:
            cur_node_start = (max_width ** (cur_height - 1) - 1) // (max_width - 1)
            cur_node_end = (max_width ** (cur_height) - 1) // (max_width - 1)
            cur_children_end = (max_width ** (cur_height + 1) - 1) // (max_width - 1)
            cur_goal = goals[:, :cur_node_end - cur_node_start, :]

            pos_embedding = self.position_embedding(cur_goal, max_width)
            pos_embedding_for_attn = pos_embedding.view(pos_embedding.size(0), -1, pos_embedding.size(-1))
            self_attn_output, _ = self.self_attn(pos_embedding_for_attn, pos_embedding_for_attn, pos_embedding_for_attn)

            candidate_hidden_states_ffn = self.candidate_ffn(candidate_hidden_states)
            # inter_attn_output, _ = self.inter_attn(query=self_attn_output,
            #                                        key=candidate_hidden_states_ffn,
            #                                        value=candidate_hidden_states_ffn,
            #                                        key_padding_mask=candidate_attn_mask)

            inter_attn_output, _ = self.inter_attn(query=self_attn_output,
                                                   key=candidate_hidden_states_ffn,
                                                   value=candidate_hidden_states_ffn,
                                                   key_padding_mask=candidate_attn_mask)
            inter_attn_output_res = self.residusal_LN(inter_attn_output + self_attn_output)
            inter_attn_output, _ = self.inter_attn1(query=inter_attn_output_res,
                                                    key=candidate_hidden_states_ffn,
                                                    value=candidate_hidden_states_ffn,
                                                    key_padding_mask=candidate_attn_mask)
            inter_attn_output = self.residusal_LN1(inter_attn_output + inter_attn_output_res)

            probilities, comparision = self.ptr_net(candidate_hidden_states, inter_attn_output, candidate_attn_mask)
            pred_tokens = probilities.argmax(dim=-1).view(batch_size, -1)

            predict = torch.cat([predict, pred_tokens], dim=1)
            classifier_input = inter_attn_output
            classifier_logits = self.format_ffn(classifier_input)
            classifier_pred = classifier_logits.argmax(dim=-1)

            classifier_mask = pred_tokens < self.num_start
            classifier_predict = classifier_pred.masked_fill(classifier_mask, 0)
            predict_format = torch.cat([predict_format, classifier_predict], dim=1)

            assert inter_attn_output.size(0) == pos_embedding.size(0)
            # assert inter_attn_output.size(1)==pos_embedding.size(1)
            goals = self.sub_goal_ffn(inter_attn_output)
            cur_height += 1

        pad_idx = word2index['PAD']
        end_idx = word2index['EOS']
        self.pad_idx = pad_idx
        self.end_idx = end_idx
        predict_mtree = self.recover_mtree(predict, predict_format, max_height, max_width, 1, 0)
        node_right, format_right, mtree_right, result_batch, predict_mtrees, target_mtrees = self.mtree_equalbatch(predict_mtree,
                                                                                                    target_mtree)
        return node_right, format_right, mtree_right, result_batch, predict_mtrees, target_mtrees

    def recover_mtree(self,
                      mtree_data,
                      mtree_format,
                      max_height,
                      max_width,
                      height,
                      layer_idx):
        if height == max_height:
            index = (max_width ** (height - 1) - 1) // (max_width - 1) + layer_idx
            mtree = {}
            mtree['data'] = mtree_data[:, index]
            mtree['format'] = mtree_format[:, index]
            mtree['children'] = []
            return mtree
        else:
            index = (max_width ** (height - 1) - 1) // (max_width - 1) + layer_idx
            mtree = {}
            mtree['data'] = mtree_data[:, index]
            mtree['format'] = mtree_format[:, index]
            mtree['children'] = []
            for i in range(max_width):
                mtree['children'].append(
                    self.recover_mtree(mtree_data, mtree_format, max_height, max_width, height + 1,
                                       layer_idx * max_width + i)
                )
            return mtree

    def mtree_equal_batch(self, mtree1_batch, mtree2_batch):
        node_right = 0
        format_right = 0
        mtree_right = 0
        result = []
        batch_size = mtree1_batch['data'].size(0)
        for i in range(batch_size):
            mtree1 = self.mtree_batch_to_mtrees(mtree1_batch, i)
            self.evaluate_mtree_pad(mtree1, self.pad_idx, self.end_idx)
            mtree1_code = self.mtree_to_code(mtree1)
            mtree2 = self.mtree_batch_to_mtrees(mtree2_batch, i)
            mtree2_code = self.mtree_to_code(mtree2)

            mtree_flag, node_flag, format_flag = self.mtree_equal_code(mtree1_code, mtree2_code)
            node_right += node_flag
            format_right += format_flag
            mtree_right += mtree_flag
            result.append({
                'node_flag': node_flag,
                'format_flag': format_flag,
                'mtree_flag': mtree_flag})
        return node_right, format_right, mtree_right, result

    def mtree_equalbatch(self, mtree1_batch, mtree2_batch):
        node_right = 0
        format_right = 0
        mtree_right = 0
        result = []
        batch_size = mtree1_batch['data'].size(0)
        predict_mtrees = []
        target_mtrees = []
        for i in range(batch_size):
            mtree1 = self.mtree_batch_to_mtrees(mtree1_batch, i)
            self.evaluate_mtree_pad(mtree1, self.pad_idx, self.end_idx)
            predict_mtrees.append(mtree1)
            mtree1_code = self.mtree_to_code(mtree1)
            mtree2 = self.mtree_batch_to_mtrees(mtree2_batch, i)
            target_mtrees.append(mtree2)
            mtree2_code = self.mtree_to_code(mtree2)

            mtree_flag, node_flag, format_flag = self.mtree_equal_code(mtree1_code, mtree2_code)
            node_right += node_flag
            format_right += format_flag
            mtree_right += mtree_flag
            result.append({
                'node_flag': node_flag,
                'format_flag': format_flag,
                'mtree_flag': mtree_flag})
        return node_right, format_right, mtree_right, result, predict_mtrees, target_mtrees

    def mtree_batch_to_mtrees(self, mtree_batch, index):
        '''
        将batch的mtree转换为单个mtree
        '''
        mtree = {}
        mtree['data'] = mtree_batch['data'][index].item()
        mtree['format'] = mtree_batch['format'][index].item()
        mtree['children'] = []
        if len(mtree_batch['children']) == 0:
            return mtree
        else:
            for child in mtree_batch['children']:
                mtree['children'].append(self.mtree_batch_to_mtrees(child, index))
            return mtree

    def evaluate_mtree_pad(self,mtree,pad_idx,end_idx,pad_all=False):
        if pad_all:
            mtree['data']=pad_idx
            mtree['format']=0
            if len(mtree['children'])==0:
                return
            else:
                for child in mtree['children']:
                    self.evaluate_mtree_pad(child,pad_idx,end_idx,True)
        else:
            data=mtree['data']
            if data==pad_idx or data>=self.num_start or data==end_idx:
                if len(mtree['children'])==0:
                    return
                else:
                    for child in mtree['children']:
                        self.evaluate_mtree_pad(child,pad_idx,end_idx,True)
            else:
                if len(mtree['children'])==0:
                    return
                else:
                    end_flag=False 
                    for child in mtree['children']:
                        if end_flag:
                            child['data']=pad_idx
                        if child['data']==end_idx:
                            end_flag=True
                        self.evaluate_mtree_pad(child,pad_idx,end_idx,False)

    def mtree_to_code(self, mtree):
        data = mtree['data']
        format = mtree['format']
        code = [{'data': data, 'format': format}]
        if len(mtree['children']) == 0:
            return [code]
        else:
            return_code = []
            for child in mtree['children']:
                child_code = self.mtree_to_code(child)
                for c in child_code:
                    return_code.append(code + c)
            return return_code

    def mtree_equal_code(self, mtree1_code, mtree2_code):
        if len(mtree1_code) != len(mtree2_code):
            return False
        mtree1_code_str = []
        for code1 in mtree1_code:
            cur_code_str = ''
            for token in code1:
                cur_token_str = ''
                cur_token_str += ('_' + str(token['data']))
                if token['data'] == self.pad_idx or token['data'] == self.end_idx:
                    cur_token_str += ('-' + str(0))
                else:
                    cur_token_str += ('-' + str(token['format']))
                cur_code_str += cur_token_str
            mtree1_code_str.append(cur_code_str)
        mtree2_code_str = []
        for code2 in mtree2_code:
            cur_code_str = ''
            for token in code2:
                cur_token_str = ''
                cur_token_str += ('_' + str(token['data']))
                if token['data'] == self.pad_idx or token['data'] == self.end_idx:
                    cur_token_str += ('-' + str(0))
                else:
                    cur_token_str += ('-' + str(token['format']))
                cur_code_str += cur_token_str
            mtree2_code_str.append(cur_code_str)

        flag = True

        for i in range(len(mtree1_code_str)):
            if mtree1_code_str[i] in mtree2_code_str:
                mtree2_code_str.remove(mtree1_code_str[i])
            else:
                flag = False
                break
        if len(mtree2_code_str) != 0:
            flag = False

        mtree1_code_node_str = []
        mtree1_code_format_str = []
        for code1 in mtree1_code:
            cur_code_node_str = ''
            cur_code_format_str = ''
            for token in code1:
                cur_code_node_str += ('_' + str(token['data']))
                if token['data'] == self.pad_idx:
                    # cur_code_str+=('_'+str(token['data'])+'-'+str(0))
                    cur_code_format_str += ('_' + str(0))
                else:
                    # cur_code_str+=('_'+str(token['data'])+'-'+str(token['format']))
                    cur_code_format_str += ('_' + str(token['format']))
            mtree1_code_node_str.append(cur_code_node_str)
            mtree1_code_format_str.append(cur_code_format_str)
        mtree2_code_node_str = []
        mtree2_code_format_str = []
        for code2 in mtree2_code:
            cur_code_node_str = ''
            cur_code_format_str = ''
            for token in code2:
                cur_code_node_str += ('_' + str(token['data']))
                if token['data'] == self.pad_idx:
                    # cur_code_str+=('_'+str(token['data'])+'-'+str(0))
                    cur_code_format_str += ('_' + str(0))
                else:
                    # cur_code_str+=('_'+str(token['data'])+'-'+str(token['format']))
                    cur_code_format_str += ('_' + str(token['format']))
            mtree2_code_node_str.append(cur_code_node_str)
            mtree2_code_format_str.append(cur_code_format_str)

        node_flag = False
        format_flag = True
        # node
        for i in range(len(mtree1_code_node_str)):
            if mtree1_code_node_str[i] in mtree2_code_node_str:
                mtree2_code_node_str.remove(mtree1_code_node_str[i])
        node_flag = len(mtree2_code_node_str) == 0

        # format
        for i in range(len(mtree1_code_format_str)):
            if mtree1_code_format_str[i] != mtree2_code_format_str[i]:
                format_flag = False
                break

        return flag, node_flag, format_flag
