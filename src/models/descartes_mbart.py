import copy
import inspect
import logging
import warnings

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import MBartModel, MBartForConditionalGeneration, MBartConfig, BeamSearchScorer, GenerationConfig, \
    LogitsProcessorList, StoppingCriteriaList, DisjunctiveConstraint, ConstrainedBeamSearchScorer, PhrasalConstraint
from transformers.generation.utils import GreedySearchOutput, SampleOutput, BeamSearchOutput, BeamSampleOutput, \
    GenerateOutput, GenerationMode
from transformers.utils.generic import ModelOutput
from transformers.models.mbart.modeling_mbart import MBartAttention, shift_tokens_right
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqModelOutput, Seq2SeqLMOutput
from typing import Union, Optional, Iterable, Callable, List, Dict, Any, Tuple

_CHECKPOINT_FOR_DOC = "facebook/mbart-large-cc25"
_CONFIG_FOR_DOC = "MBartConfig"
_TOKENIZER_FOR_DOC = "MBartTokenizer"


class MBartModelDescartes(MBartModel):
    def __init__(self, config:MBartConfig):
        super().__init__(config)
        self.expand = nn.Linear(1, config.d_model)
        self.mapping = MBartAttention(config.d_model, config.decoder_attention_heads, dropout=config.attention_dropout,
                                      is_decoder=True)
        self.norm = nn.LayerNorm(config.d_model)

        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.embed_dim = config.d_model
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

        self.graph_mapping = nn.Linear(config.graph_embd_length, config.d_model)
        self.bert_mapping = nn.Linear(768, config.d_model)
        self.post_init()



    def get_expand(self):
        return self.expand

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            target_lang=None,
            main_lang=None,
            graph_embeddings=None,
            bert_outputs=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # different to other models, MBart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(input_ids, self.config.pad_token_id)

        assert (target_lang is not None)
        assert (main_lang is not None)

        lang_out = torch.ones((attention_mask[main_lang].shape[0], 1, 1), device=attention_mask[main_lang].device)
        lang_out = self.expand(lang_out)

        if encoder_outputs is None:
            encoder_outputs = {}
            for lang, input_ids_val in input_ids.items():
                if input_ids_val is not None:
                    encoder_outputs_val = self.encoder(
                        input_ids=input_ids_val,
                        attention_mask=attention_mask[lang],
                        head_mask=head_mask,
                        inputs_embeds=inputs_embeds,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=return_dict,
                    )
                else:
                    encoder_outputs_val = BaseModelOutput(
                        last_hidden_state=lang_out,
                        hidden_states=None,
                        attentions=None,
                    )
                encoder_outputs[lang] = encoder_outputs_val
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=False
        elif return_dict:
            for key, encoder_outputs_val in encoder_outputs.items():
                if not isinstance(encoder_outputs_val, BaseModelOutput):
                    encoder_outputs_val = BaseModelOutput(
                        last_hidden_state=encoder_outputs_val[0],
                        hidden_states=encoder_outputs_val[1] if len(encoder_outputs_val) > 1 else None,
                        attentions=encoder_outputs_val[2] if len(encoder_outputs_val) > 2 else None,
                    )
                    encoder_outputs[key] = encoder_outputs_val

        enc_outputs = None
        attn_mask = attention_mask[main_lang]
        if len(encoder_outputs) != 0:
            key, query, mask = None, None, None
            enc_outputs_list = []

            query_main = encoder_outputs[main_lang][0]

            for lang, key in encoder_outputs.items():
                key = key[0]
                mask = attention_mask[lang] #TODO check
                query = query_main


                enc_outputs, att_weight, _ = self.mapping(hidden_states=query, key_value_states=key,
                                                          output_attentions=output_attentions)
                enc_outputs = enc_outputs + query
                enc_outputs = self.norm(enc_outputs)

                residual = enc_outputs
                enc_outputs = self.activation_fn(self.fc1(enc_outputs))
                enc_outputs = F.dropout(enc_outputs, p=self.activation_dropout, training=self.training)
                enc_outputs = self.fc2(enc_outputs)
                enc_outputs = F.dropout(enc_outputs, p=self.dropout, training=self.training)
                enc_outputs = residual + enc_outputs
                enc_outputs = self.final_layer_norm(enc_outputs)

                enc_outputs_list.append(enc_outputs)

            enc_outputs = torch.mean(torch.stack(enc_outputs_list), dim=0)

        # add graph embedding
        if graph_embeddings is not None:
            graph_embeddings_mapped = self.graph_mapping(graph_embeddings)
            graph_embeddings_mapped = torch.reshape(graph_embeddings_mapped, shape=(
            graph_embeddings_mapped.shape[0], 1, graph_embeddings_mapped.shape[1]))
            if enc_outputs is None:
                enc_outputs = graph_embeddings_mapped
                attn_mask = torch.ones((attn_mask.shape[0], 1), device=enc_outputs.device)
            else:
                enc_outputs = torch.cat((enc_outputs, graph_embeddings_mapped), 1)
                new_mask_column = torch.ones((attn_mask.shape[0], 1), device=enc_outputs.device)
                attn_mask = torch.cat((attn_mask, new_mask_column), dim=1)

        # adding summary embedding
        if bert_outputs is not None:
            bert_outputs = self.bert_mapping(bert_outputs)
            bert_outputs = torch.reshape(bert_outputs, shape=(bert_outputs.shape[0], 1, bert_outputs.shape[1]))
            if enc_outputs is None:
                enc_outputs = bert_outputs
                attn_mask = torch.ones((attn_mask.shape[0], 1), device=enc_outputs.device)
            else:
                enc_outputs = torch.cat((enc_outputs, bert_outputs), 1)
                new_mask_column = torch.ones((attn_mask.shape[0], 1), device=enc_outputs.device)
                attn_mask = torch.cat((attn_mask, new_mask_column), dim=1)

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=enc_outputs,
            encoder_attention_mask=attn_mask,
            head_mask=decoder_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        if len(encoder_outputs) != 0:
            enc_last_hidden_state = encoder_outputs[main_lang].last_hidden_state
            enc_hidden_states = encoder_outputs[main_lang].hidden_states
            enc_attentions = encoder_outputs[main_lang].attentions
        else:
            enc_last_hidden_state = None
            enc_hidden_states = None
            enc_attentions = None
        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=enc_last_hidden_state,
            encoder_hidden_states=enc_hidden_states,
            encoder_attentions=enc_attentions,
        )


class MBartForConditionalGenerationDescartes(MBartForConditionalGeneration):
    def __init__(self, config: MBartConfig):
        super().__init__(config)
        self.model = MBartModelDescartes(config)
        self.model_bert = None
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_expand(self):
        return self.model.get_expand()

    def get_model_bert(self):
        return self.model_bert

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            target_lang=None,
            main_lang=None,
            graph_embeddings=None,
            bert_inputs=None,
            bert_outputs=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            labels = labels[target_lang[0:2]]

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id)

        bert_outputs_list = []
        if (bert_outputs is None) and (self.model_bert is not None):
            for lang, bert_in in bert_inputs.items():
                bert_outs = self.model_bert(**bert_in)
                bert_outs = torch.mean(bert_outs.last_hidden_state, dim=1)
                bert_outputs_list.append(bert_outs)
            if len(bert_outputs_list) == 0:
                bert_outputs = torch.zeros((1, 768), device=attention_mask[target_lang[0:2]].device)
            else:
                bert_outputs = torch.mean(torch.stack(bert_outputs_list), dim=0)


        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            target_lang=target_lang,
            main_lang=main_lang,
            graph_embeddings=graph_embeddings,
            bert_outputs=bert_outputs,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
            self, decoder_input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            "target_lang": kwargs["target_lang"],
            "main_lang": kwargs["main_lang"],
            "graph_embeddings": kwargs["graph_embeddings"],
            "bert_inputs": kwargs["bert_inputs"],
            "bert_outputs": kwargs["bert_outputs"],
        }

    def _prepare_attention_mask_for_generation(
            self, target_lang, input_ids: torch.Tensor, pad_token_id: int, eos_token_id: int
    ):
        attention_mask = {}
        for lang in list(input_ids.keys()):
            is_pad_token_in_inputs_ids = (pad_token_id is not None) and (input_ids[lang] is not None) and (
                        pad_token_id in input_ids[lang])
            is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (
                    (eos_token_id is not None) and (pad_token_id != eos_token_id)
            )
            if is_pad_token_in_inputs_ids and is_pad_token_not_equal_to_eos_token_id:
                attention_mask[lang] = input_ids[lang].ne(pad_token_id).long()
            elif input_ids[lang] is not None:
                attention_mask[lang] = input_ids[lang].new_ones(input_ids[lang].shape)
            else:
                attention_mask[lang] = input_ids[target_lang].new_ones((input_ids[target_lang].shape[0], 1))

        return attention_mask

    def _prepare_encoder_decoder_kwargs_for_generation(
            self, target_lang, input_ids: torch.LongTensor, baseline, mask_text, model_kwargs
    ) -> Dict[str, Any]:
        if baseline:
            if "encoder_outputs" not in model_kwargs:
                # retrieve encoder hidden states
                encoder = self.get_encoder()
                encoder_kwargs = {
                    argument: value for argument, value in model_kwargs.items() if not argument.startswith("decoder_")
                }
                attention_mask = encoder_kwargs.pop("attention_mask")
                encoder_kwargs.pop("target_lang")
                attention_mask = attention_mask[target_lang]
                encoder_kwargs.pop("graph_embeddings")
                encoder_kwargs.pop("bert_inputs")

                model_kwargs["encoder_outputs"]: ModelOutput = encoder(input_ids[target_lang],
                                                                       attention_mask=attention_mask, return_dict=True,
                                                                       **encoder_kwargs)
            return model_kwargs

        else:
            encoder = self.get_encoder()
            expand = self.get_expand()
            if not mask_text:
                encoder_kwargs = {
                    argument: value for argument, value in model_kwargs.items() if not argument.startswith("decoder_")
                }
                attention_mask = encoder_kwargs.pop("attention_mask")
                encoder_kwargs.pop("target_lang")
                encoder_kwargs.pop("graph_embeddings")
                encoder_kwargs.pop("bert_inputs")
                encoder_kwargs.pop("use_cache")

                lang_out = torch.ones((attention_mask[target_lang].shape[0], 1, 1),
                                      device=attention_mask[target_lang].device)
                lang_out = expand(lang_out)
                encoder_outputs = {}

                for lang, inputs in input_ids.items():
                    if inputs is not None:
                        enc_out = encoder(inputs, attention_mask=attention_mask[lang], return_dict=True,
                                          **encoder_kwargs)
                    else:
                        enc_out = BaseModelOutput(
                            last_hidden_state=lang_out,
                            hidden_states=None,
                            attentions=None,
                        )
                    encoder_outputs[lang] = enc_out
                model_kwargs["encoder_outputs"] = encoder_outputs
            else:
                model_kwargs["encoder_outputs"] = {}
            return model_kwargs

    def _prepare_decoder_input_ids_for_generation(
            self, target_lang, input_ids: torch.LongTensor, model_kwargs, decoder_start_token_id: int = None,
            bos_token_id: int = None
    ) -> torch.LongTensor:
        if "decoder_input_ids" in model_kwargs:
            return model_kwargs["decoder_input_ids"]

        decoder_start_token_id = self._get_decoder_start_token_id(decoder_start_token_id, bos_token_id)

        shape = input_ids[target_lang].shape[0]
        t = input_ids[target_lang].dtype
        device = input_ids[target_lang].device

        decoder_input_ids = (
                torch.ones((shape, 1), dtype=t, device=device)
                * decoder_start_token_id
        )
        return decoder_input_ids

    def _prepare_bert_outputs(self, target_lang, bert_inputs, model_kwargs):
        bert_outputs_list = []
        for lang, bert_in in bert_inputs.items():
            bert_outs = self.model_bert(**bert_in)
            bert_outs = torch.mean(bert_outs.last_hidden_state, dim=1)
            bert_outputs_list.append(bert_outs)
        if len(bert_outputs_list) == 0:
            bert_outputs = torch.zeros((1, 768), device=model_kwargs["attention_mask"][target_lang[0:2]].device)
        else:
            bert_outputs = torch.mean(torch.stack(bert_outputs_list), dim=0)
        return bert_outputs

    @staticmethod
    def _expand_inputs_for_generation(
            input_ids: torch.LongTensor,
            expand_size: int = 1,
            is_encoder_decoder: bool = False,
            attention_mask: torch.LongTensor = None,
            encoder_outputs: ModelOutput = None,
            baseline=False,
            **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        # target_lang = model_kwargs["target_lang"][0:2]
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )

        input_ids = input_ids.index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(0, expanded_return_idx)


        if attention_mask is not None:
            attention_mask_expanded = {}
            for lang, mask in attention_mask.items():
                if mask is not None:
                    mask = mask.index_select(0, expanded_return_idx)
                attention_mask_expanded[lang] = mask
            model_kwargs["attention_mask"] = attention_mask_expanded

        if is_encoder_decoder:
            assert encoder_outputs is not None
            if len(encoder_outputs) != 0:
                if baseline:
                    encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
                        0, expanded_return_idx.to(encoder_outputs.last_hidden_state.device)
                    )
                    model_kwargs["encoder_outputs"] = encoder_outputs
                else:
                    for lang, enc_out in encoder_outputs.items():
                        enc_out["last_hidden_state"] = enc_out.last_hidden_state.index_select(0, expanded_return_idx)
                        encoder_outputs[lang] = enc_out
                    model_kwargs["encoder_outputs"] = encoder_outputs
            else:
                model_kwargs["encoder_outputs"] = encoder_outputs

        graph_embds = model_kwargs["graph_embeddings"]
        if graph_embds is not None:
            graph_embds = graph_embds.index_select(0, expanded_return_idx)
            model_kwargs["graph_embeddings"] = graph_embds

        bert_outs = model_kwargs["bert_outputs"]
        if bert_outs is not None:
            bert_outs = bert_outs.index_select(0, expanded_return_idx)
            model_kwargs["bert_outputs"] = bert_outs

        return input_ids, model_kwargs

    @torch.no_grad()
    def generate(
            self,
            inputs: Optional[torch.Tensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
            synced_gpus: Optional[bool] = None,
            assistant_model: Optional["PreTrainedModel"] = None,
            streamer: Optional["BaseStreamer"] = None,
            negative_prompt_ids: Optional[torch.Tensor] = None,
            negative_prompt_attention_mask: Optional[torch.Tensor] = None,
            target_lang: Optional[str] = None,
            main_lang: Optional[str] = None,
            baseline=False,
            mask_text=False,
            **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        self._validate_model_class()

        if generation_config is None:
            # legacy: users may modify the model configuration to control generation -- update the generation config
            # model attribute accordingly, if it was created from the model config
            if self.generation_config._from_model_config:
                new_generation_config = GenerationConfig.from_model_config(self.config)
                if new_generation_config != self.generation_config:
                    warnings.warn(
                        "You have modified the pretrained model configuration to control generation. This is a"
                        " deprecated strategy to control generation and will be removed soon, in a future version."
                        " Please use a generation configuration file (see"
                        " https://huggingface.co/docs/transformers/main_classes/text_generation )"
                    )
                    self.generation_config = new_generation_config
            generation_config = self.generation_config
            generation_config = copy.deepcopy(generation_config)
            model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs
            generation_config.validate()
            self._validate_model_kwargs(model_kwargs.copy())


        # 2. Set generation parameters if not already defined
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
            eos_token_id = generation_config.eos_token_id
            if isinstance(eos_token_id, list):
                eos_token_id = eos_token_id[0]
            generation_config.pad_token_id = eos_token_id

        # 3. Define model inputs
        # inputs_tensor has to be defined
        # model_input_name is defined if model-specific keyword input is passed
        # otherwise model_input_name is None
        # all model-specific keyword inputs are removed from `model_kwargs`
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        representative_tensor = None
        for lang, t in inputs_tensor.items():
            if isinstance(t, torch.Tensor):
                representative_tensor = lang
                break

        batch_size = inputs_tensor[representative_tensor].shape[0]

        model_kwargs["target_lang"] = target_lang
        target_lang = target_lang[0:2]
        tgt = None
        if main_lang is None:
            tgt = target_lang
        else:
            tgt = main_lang

        # 4. Define other model kwargs
        model_kwargs["output_attentions"] = generation_config.output_attentions
        model_kwargs["output_hidden_states"] = generation_config.output_hidden_states
        # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
        # generating the first new token or not, and we only want to use the embeddings for the first new token)
        if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
            model_kwargs["use_cache"] = True
        else:
            model_kwargs["use_cache"] = generation_config.use_cache

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs

        if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                tgt, inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
            )

        if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created
            # and added to `model_kwargs`
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                tgt, inputs_tensor, baseline, mask_text, model_kwargs
            )

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        if self.config.is_encoder_decoder:
            input_ids = self._prepare_decoder_input_ids_for_generation(
                    tgt, inputs_tensor, model_kwargs, decoder_start_token_id=generation_config.decoder_start_token_id, bos_token_id=generation_config.bos_token_id
                )
        else:
            input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        # Prepare bert inputs and outputs
        bert_inputs = model_kwargs["bert_inputs"]
        if bert_inputs is not None:
            bert_outputs = self._prepare_bert_outputs(tgt, bert_inputs, model_kwargs)
            model_kwargs["bert_outputs"] = bert_outputs

        else:
            model_kwargs["bert_outputs"] = None

        if main_lang is None:
            model_kwargs["main_lang"] = target_lang
        else:
            model_kwargs["main_lang"] = main_lang

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        if generation_config.max_new_tokens is not None:
            generation_config.max_length = generation_config.max_new_tokens + input_ids_length
        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        # 7. determine generation mode
        generation_mode = self._get_generation_mode(generation_config, assistant_model)

        if self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )

        # 8. prepare distribution pre_processing samplers
        logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            model_kwargs=model_kwargs,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )

        # 9. prepare stopping criteria
        stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria
        )

        # 10. go into different generation modes
        if generation_mode == GenerationMode.ASSISTED_GENERATION:
            if generation_config.num_return_sequences > 1:
                raise ValueError(
                    "num_return_sequences has to be 1 when doing assisted generate, "
                    f"but is {generation_config.num_return_sequences}."
                )
            if batch_size > 1:
                raise ValueError("assisted generate is only supported for batch_size = 1")
            if not model_kwargs["use_cache"]:
                raise ValueError("assisted generate requires `use_cache=True`")

            # 11. If the assistant model is an encoder-decoder, prepare its encoder outputs
            if assistant_model.config.is_encoder_decoder:
                assistant_model_kwargs = copy.deepcopy(model_kwargs)
                inputs_tensor, model_input_name, assistant_model_kwargs = assistant_model._prepare_model_inputs(
                    inputs_tensor, assistant_model.generation_config.bos_token_id, assistant_model_kwargs
                )
                assistant_model_kwargs = assistant_model._prepare_encoder_decoder_kwargs_for_generation(
                    inputs_tensor, assistant_model_kwargs, model_input_name
                )
                model_kwargs["assistant_encoder_outputs"] = assistant_model_kwargs["encoder_outputs"]

            # 12. run assisted generate
            return self.assisted_decoding(
                input_ids,
                assistant_model=assistant_model,
                do_sample=generation_config.do_sample,
                logits_processor=logits_processor,
                logits_warper=self._get_logits_warper(generation_config) if generation_config.do_sample else None,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )
        if generation_mode == GenerationMode.GREEDY_SEARCH:
            # 11. run greedy search
            return self.greedy_search(
                input_ids,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.CONTRASTIVE_SEARCH:
            if not model_kwargs["use_cache"]:
                raise ValueError("Contrastive search requires `use_cache=True`")

            return self.contrastive_search(
                input_ids,
                top_k=generation_config.top_k,
                penalty_alpha=generation_config.penalty_alpha,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                streamer=streamer,
                sequential=generation_config.low_memory,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.SAMPLE:
            # 11. prepare logits warper
            logits_warper = self._get_logits_warper(generation_config)

            # 12. expand input_ids with `num_return_sequences` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids,
                expand_size=generation_config.num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                baseline=baseline,
                **model_kwargs,
            )

            # 13. run sample
            return self.sample(
                input_ids,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.BEAM_SEARCH:
            # 11. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor[representative_tensor].device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                max_length=generation_config.max_length,
            )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids, expand_size=generation_config.num_beams, is_encoder_decoder=self.config.is_encoder_decoder, baseline=baseline,
                **model_kwargs
            )
            # 13. run beam search



            return self.beam_search(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.BEAM_SAMPLE:
            # 11. prepare logits warper
            logits_warper = self._get_logits_warper(generation_config)

            # 12. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                max_length=generation_config.max_length,
            )

            # 13. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                baseline=baseline,
                **model_kwargs,
            )

            # 14. run beam sample
            return self.beam_sample(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.GROUP_BEAM_SEARCH:
            # 11. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                num_beam_groups=generation_config.num_beam_groups,
                max_length=generation_config.max_length,
            )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids,
                expand_size=generation_config.num_beams * generation_config.num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                baseline=baseline,
                **model_kwargs,
            )
            # 13. run beam search
            return self.group_beam_search(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.CONSTRAINED_BEAM_SEARCH:
            final_constraints = []
            if generation_config.constraints is not None:
                final_constraints = generation_config.constraints

            if generation_config.force_words_ids is not None:

                def typeerror():
                    raise ValueError(
                        "`force_words_ids` has to either be a `List[List[List[int]]]` or `List[List[int]]`"
                        f"of positive integers, but is {generation_config.force_words_ids}."
                    )

                if (
                        not isinstance(generation_config.force_words_ids, list)
                        or len(generation_config.force_words_ids) == 0
                ):
                    typeerror()

                for word_ids in generation_config.force_words_ids:
                    if isinstance(word_ids[0], list):
                        if not isinstance(word_ids, list) or len(word_ids) == 0:
                            typeerror()
                        if any(not isinstance(token_ids, list) for token_ids in word_ids):
                            typeerror()
                        if any(
                                any((not isinstance(token_id, int) or token_id < 0) for token_id in token_ids)
                                for token_ids in word_ids
                        ):
                            typeerror()

                        constraint = DisjunctiveConstraint(word_ids)
                    else:
                        if not isinstance(word_ids, list) or len(word_ids) == 0:
                            typeerror()
                        if any((not isinstance(token_id, int) or token_id < 0) for token_id in word_ids):
                            typeerror()

                        constraint = PhrasalConstraint(word_ids)
                    final_constraints.append(constraint)

            # 11. prepare beam search scorer
            constrained_beam_scorer = ConstrainedBeamSearchScorer(
                constraints=final_constraints,
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                max_length=generation_config.max_length,
            )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                baseline=baseline,
                **model_kwargs,
            )
            # 13. run beam search
            return self.constrained_beam_search(
                input_ids,
                constrained_beam_scorer=constrained_beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )



