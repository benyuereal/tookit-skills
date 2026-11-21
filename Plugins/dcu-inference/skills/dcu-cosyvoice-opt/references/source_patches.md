<!-- 溯源: 内部文档 file_id=<internal-file-id> (CosyVoice核心代码深度解读与性能优化实践, 作者 <internal-author>) -->
<!-- 说明: 以下为文档摘录的原生/修改片段, 仅供溯源与对照, 非独立可运行文件; 完整可运行请参见上游 FunAudioLLM/CosyVoice -->

## flow.py —— MaskedDiffWithXvec / CausalMaskedDiffWithXvec（含 n_timesteps=5 修改）

```python

class MaskedDiffWithXvec(torch.nn.Module):  
    def __init__(self,  
                 input_size: int = 512,  
                 output_size: int = 80,  
                 spk_embed_dim: int = 192,  
                 output_type: str = "mel",  
                 vocab_size: int = 4096,  
                 input_frame_rate: int = 50,  
                 only_mask_loss: bool = True,  
                 encoder: torch.nn.Module = None,  
                 length_regulator: torch.nn.Module = None,  
                 decoder: torch.nn.Module = None,  
                 decoder_conf: Dict = {'in_channels': 240, 'out_channel': 80, 'spk_emb_dim': 80, 'n_spks': 1,  
                                       'cfm_params': DictConfig({'sigma_min': 1e-06, 'solver': 'euler', 't_scheduler': 'cosine',  
                                                                 'training_cfg_rate': 0.2, 'inference_cfg_rate': 0.7, 'reg_loss_type': 'l1'}),  
                                       'decoder_params': {'channels': [256, 256], 'dropout': 0.0, 'attention_head_dim': 64,  
                                                          'n_blocks': 4, 'num_mid_blocks': 12, 'num_heads': 8, 'act_fn': 'gelu'}},  
                 mel_feat_conf: Dict = {'n_fft': 1024, 'num_mels': 80, 'sampling_rate': 22050,  
                                        'hop_size': 256, 'win_size': 1024, 'fmin': 0, 'fmax': 8000}):  
        super().__init__()  
        self.input_size = input_size  
        self.output_size = output_size  
        self.decoder_conf = decoder_conf  
        self.mel_feat_conf = mel_feat_conf  
        self.vocab_size = vocab_size  
        self.output_type = output_type  
        self.input_frame_rate = input_frame_rate  
        logging.info(f"input frame rate={self.input_frame_rate}")  
        self.input_embedding = nn.Embedding(vocab_size, input_size)  
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)  
        self.encoder = encoder  
        self.encoder_proj = torch.nn.Linear(self.encoder.output_size(), output_size)  
        self.decoder = decoder  
        self.length_regulator = length_regulator  
        self.only_mask_loss = only_mask_loss  
  
    def forward(  
            self,  
            batch: dict,  
            device: torch.device,  
    ) -> Dict[str, Optional[torch.Tensor]]:  
        token = batch['speech_token'].to(device)  
        token_len = batch['speech_token_len'].to(device)  
        feat = batch['speech_feat'].to(device)  
        feat_len = batch['speech_feat_len'].to(device)  
        embedding = batch['embedding'].to(device)  
  
        # xvec projection  
        embedding = F.normalize(embedding, dim=1)  
        embedding = self.spk_embed_affine_layer(embedding)  
  
        # concat text and prompt_text  
        mask = (~make_pad_mask(token_len)).float().unsqueeze(-1).to(device)  
        token = self.input_embedding(torch.clamp(token, min=0)) * mask  
  
        # text encode  
        h, h_lengths = self.encoder(token, token_len)  
        h = self.encoder_proj(h)  
        h, h_lengths = self.length_regulator(h, feat_len)  
  
        # get conditions  
        conds = torch.zeros(feat.shape, device=token.device)  
        for i, j in enumerate(feat_len):  
            if random.random() < 0.5:  
                continue  
            index = random.randint(0, int(0.3 * j))  
            conds[i, :index] = feat[i, :index]  
        conds = conds.transpose(1, 2)  
  
        mask = (~make_pad_mask(feat_len)).to(h)  
        # NOTE this is unnecessary, feat/h already same shape  
        loss, _ = self.decoder.compute_loss(  
            feat.transpose(1, 2).contiguous(),  
            mask.unsqueeze(1),  
            h.transpose(1, 2).contiguous(),  
            embedding,  
            cond=conds  
        )  
        return {'loss': loss}  
  
    @torch.inference_mode()  
    def inference(self,  
                  token,  
                  token_len,  
                  prompt_token,  
                  prompt_token_len,  
                  prompt_feat,  
                  prompt_feat_len,  
                  embedding,  
                  flow_cache):  
        assert token.shape[0] == 1  
        # xvec projection  
        embedding = F.normalize(embedding, dim=1)  
        embedding = self.spk_embed_affine_layer(embedding)  
  
        # concat speech token and prompt speech token  
        token_len1, token_len2 = prompt_token.shape[1], token.shape[1]  
        token, token_len = torch.concat([prompt_token, token], dim=1), prompt_token_len + token_len  
        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)  
        token = self.input_embedding(torch.clamp(token, min=0)) * mask  
  
        # text encode  
        h, h_lengths = self.encoder(token, token_len)  
        h = self.encoder_proj(h)  
        mel_len1, mel_len2 = prompt_feat.shape[1], int(token_len2 / self.input_frame_rate * 22050 / 256)  
        h, h_lengths = self.length_regulator.inference(h[:, :token_len1], h[:, token_len1:], mel_len1, mel_len2, self.input_frame_rate)  
  
        # get conditions  
        conds = torch.zeros([1, mel_len1 + mel_len2, self.output_size], device=token.device).to(h.dtype)  
        conds[:, :mel_len1] = prompt_feat  
        conds = conds.transpose(1, 2)  
  
        mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h)  
        feat, flow_cache = self.decoder(  
            mu=h.transpose(1, 2).contiguous(),  
            mask=mask.unsqueeze(1),  
            spks=embedding,  
            cond=conds,  
            n_timesteps=5,  
            prompt_len=mel_len1,  
            cache=flow_cache  
        )  
        feat = feat[:, :, mel_len1:]  
        assert feat.shape[2] == mel_len2  
        return feat.float(), flow_cache  
  
class CausalMaskedDiffWithXvec(torch.nn.Module):  
    def __init__(self,  
                 input_size: int = 512,  
                 output_size: int = 80,  
                 spk_embed_dim: int = 192,  
                 output_type: str = "mel",  
                 vocab_size: int = 4096,  
                 input_frame_rate: int = 50,  
                 only_mask_loss: bool = True,  
                 token_mel_ratio: int = 2,  
                 pre_lookahead_len: int = 3,  
                 encoder: torch.nn.Module = None,  
                 decoder: torch.nn.Module = None,  
                 decoder_conf: Dict = {'in_channels': 240, 'out_channel': 80, 'spk_emb_dim': 80, 'n_spks': 1,  
                                       'cfm_params': DictConfig({'sigma_min': 1e-06, 'solver': 'euler', 't_scheduler': 'cosine',  
                                                                 'training_cfg_rate': 0.2, 'inference_cfg_rate': 0.7, 'reg_loss_type': 'l1'}),  
                                       'decoder_params': {'channels': [256, 256], 'dropout': 0.0, 'attention_head_dim': 64,  
                                                          'n_blocks': 4, 'num_mid_blocks': 12, 'num_heads': 8, 'act_fn': 'gelu'}},  
                 mel_feat_conf: Dict = {'n_fft': 1024, 'num_mels': 80, 'sampling_rate': 22050,  
                                        'hop_size': 256, 'win_size': 1024, 'fmin': 0, 'fmax': 8000}):  
        super().__init__()  
        self.input_size = input_size  
        self.output_size = output_size  
        self.decoder_conf = decoder_conf  
        self.mel_feat_conf = mel_feat_conf  
        self.vocab_size = vocab_size  
        self.output_type = output_type  
        self.input_frame_rate = input_frame_rate  
        logging.info(f"input frame rate={self.input_frame_rate}")  
        self.input_embedding = nn.Embedding(vocab_size, input_size)  
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)  
        self.encoder = encoder  
        self.encoder_proj = torch.nn.Linear(self.encoder.output_size(), output_size)  
        self.decoder = decoder  
        self.only_mask_loss = only_mask_loss  
        self.token_mel_ratio = token_mel_ratio  
        self.pre_lookahead_len = pre_lookahead_len  
  
    def forward(  
            self,  
            batch: dict,  
            device: torch.device,  
    ) -> Dict[str, Optional[torch.Tensor]]:  
        token = batch['speech_token'].to(device)  
        token_len = batch['speech_token_len'].to(device)  
        feat = batch['speech_feat'].to(device)  
        feat_len = batch['speech_feat_len'].to(device)  
        embedding = batch['embedding'].to(device)  
  
        # NOTE unified training, static_chunk_size > 0 or = 0  
        streaming = True if random.random() < 0.5 else False  
  
        # xvec projection  
        embedding = F.normalize(embedding, dim=1)  
        embedding = self.spk_embed_affine_layer(embedding)  
  
        # concat text and prompt_text  
        mask = (~make_pad_mask(token_len)).float().unsqueeze(-1).to(device)  
        token = self.input_embedding(torch.clamp(token, min=0)) * mask  
  
        # text encode  
        h, h_lengths = self.encoder(token, token_len, streaming=streaming)  
        h = self.encoder_proj(h)  
  
        # get conditions  
        conds = torch.zeros(feat.shape, device=token.device)  
        for i, j in enumerate(feat_len):  
            if random.random() < 0.5:  
                continue  
            index = random.randint(0, int(0.3 * j))  
            conds[i, :index] = feat[i, :index]  
        conds = conds.transpose(1, 2)  
  
        mask = (~make_pad_mask(h_lengths.sum(dim=-1).squeeze(dim=1))).to(h)  
        loss, _ = self.decoder.compute_loss(  
            feat.transpose(1, 2).contiguous(),  
            mask.unsqueeze(1),  
            h.transpose(1, 2).contiguous(),  
            embedding,  
            cond=conds,  
            streaming=streaming,  
        )  
        return {'loss': loss}  
  
    @torch.inference_mode()  
    def inference(self,  
                  token,  
                  token_len,  
                  prompt_token,  
                  prompt_token_len,  
                  prompt_feat,  
                  prompt_feat_len,  
                  embedding,  
                  streaming,  
                  finalize):  
        assert token.shape[0] == 1  
        # xvec projection  
        embedding = F.normalize(embedding, dim=1)  
        embedding = self.spk_embed_affine_layer(embedding)  
  
        # concat text and prompt_text  
        token, token_len = torch.concat([prompt_token, token], dim=1), prompt_token_len + token_len  
        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)  
        token = self.input_embedding(torch.clamp(token, min=0)) * mask  
  
        # text encode  
        if finalize is True:  
            h, h_lengths = self.encoder(token, token_len, streaming=streaming)  
        else:  
            token, context = token[:, :-self.pre_lookahead_len], token[:, -self.pre_lookahead_len:]  
            h, h_lengths = self.encoder(token, token_len, context=context, streaming=streaming)  
        mel_len1, mel_len2 = prompt_feat.shape[1], h.shape[1] - prompt_feat.shape[1]  
        h = self.encoder_proj(h)  
  
        # get conditions  
        conds = torch.zeros([1, mel_len1 + mel_len2, self.output_size], device=token.device).to(h.dtype)  
        conds[:, :mel_len1] = prompt_feat  
        conds = conds.transpose(1, 2)  
  
        mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h)  
        feat, _ = self.decoder(  
            mu=h.transpose(1, 2).contiguous(),  
            mask=mask.unsqueeze(1),  
            spks=embedding,  
            cond=conds,  
            n_timesteps=5,  
            streaming=streaming  
        )  
        feat = feat[:, :, mel_len1:]  
        assert feat.shape[2] == mel_len2  
        return feat.float(), None  
  
```

## model.py —— CosyVoiceModel 类（Opt3 vLLM 显存优化相关上下文）

```python

class CosyVoiceModel:  
  
    def __init__(self,  
                 llm: torch.nn.Module,  
                 flow: torch.nn.Module,  
                 hift: torch.nn.Module,  
                 fp16: bool = False):  
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
        self.llm = llm  
        self.flow = flow  
        self.hift = hift  
        self.fp16 = fp16  
        if self.fp16 is True:  
            self.llm.half()  
            self.flow.half()  
        self.token_min_hop_len = 2 * self.flow.input_frame_rate  
        self.token_max_hop_len = 4 * self.flow.input_frame_rate  
        self.token_overlap_len = 20  
        # mel fade in out  
        self.mel_overlap_len = int(self.token_overlap_len / self.flow.input_frame_rate * 22050 / 256)  
        self.mel_window = np.hamming(2 * self.mel_overlap_len)  
        # hift cache  
        self.mel_cache_len = 20  
        self.source_cache_len = int(self.mel_cache_len * 256)  
        # speech fade in out  
        self.speech_window = np.hamming(2 * self.source_cache_len)  
        # rtf and decoding related  
        self.stream_scale_factor = 1  
        assert self.stream_scale_factor >= 1, 'stream_scale_factor should be greater than 1, change it according to your actual rtf'  
        self.llm_context = torch.cuda.stream(torch.cuda.Stream(self.device)) if torch.cuda.is_available() else nullcontext()  
        self.lock = threading.Lock()  
        # dict used to store session related variable  
        self.tts_speech_token_dict = {}  
        self.llm_end_dict = {}  
        self.mel_overlap_dict = {}  
        self.flow_cache_dict = {}  
        self.hift_cache_dict = {}  
  
    def load(self, llm_model, flow_model, hift_model):  
        self.llm.load_state_dict(torch.load(llm_model, map_location=self.device), strict=True)  
        self.llm.to(self.device).eval()  
        self.flow.load_state_dict(torch.load(flow_model, map_location=self.device), strict=True)  
        self.flow.to(self.device).eval()  
        # in case hift_model is a hifigan model  
        hift_state_dict = {k.replace('generator.', ''): v for k, v in torch.load(hift_model, map_location=self.device).items()}  
        self.hift.load_state_dict(hift_state_dict, strict=True)  
        self.hift.to(self.device).eval()  
  
    def load_jit(self, llm_text_encoder_model, llm_llm_model, flow_encoder_model):  
        llm_text_encoder = torch.jit.load(llm_text_encoder_model, map_location=self.device)  
        self.llm.text_encoder = llm_text_encoder  
        llm_llm = torch.jit.load(llm_llm_model, map_location=self.device)  
        self.llm.llm = llm_llm  
        flow_encoder = torch.jit.load(flow_encoder_model, map_location=self.device)  
        self.flow.encoder = flow_encoder  
  
    def load_trt(self, flow_decoder_estimator_model, flow_decoder_onnx_model, trt_concurrent, fp16):  
        assert torch.cuda.is_available(), 'tensorrt only supports gpu!'  
        if not os.path.exists(flow_decoder_estimator_model) or os.path.getsize(flow_decoder_estimator_model) == 0:  
            convert_onnx_to_trt(flow_decoder_estimator_model, self.get_trt_kwargs(), flow_decoder_onnx_model, fp16)  
        del self.flow.decoder.estimator  
        import tensorrt as trt  
        with open(flow_decoder_estimator_model, 'rb') as f:  
            estimator_engine = trt.Runtime(trt.Logger(trt.Logger.INFO)).deserialize_cuda_engine(f.read())  
        assert estimator_engine is not None, 'failed to load trt {}'.format(flow_decoder_estimator_model)  
        self.flow.decoder.estimator = TrtContextWrapper(estimator_engine, trt_concurrent=trt_concurrent, device=self.device)  
  
    def get_trt_kwargs(self):  
        min_shape = [(2, 80, 4), (2, 1, 4), (2, 80, 4), (2, 80, 4)]  
        opt_shape = [(2, 80, 500), (2, 1, 500), (2, 80, 500), (2, 80, 500)]  
        max_shape = [(2, 80, 3000), (2, 1, 3000), (2, 80, 3000), (2, 80, 3000)]  
        input_names = ["x", "mask", "mu", "cond"]  
        return {'min_shape': min_shape, 'opt_shape': opt_shape, 'max_shape': max_shape, 'input_names': input_names}  
  
    def llm_job(self, text, prompt_text, llm_prompt_speech_token, llm_embedding, uuid):  
        with self.llm_context, torch.cuda.amp.autocast(self.fp16 is True and hasattr(self.llm, 'vllm') is False):  
            if isinstance(text, Generator):  
                assert isinstance(self, CosyVoice2Model) and not hasattr(self.llm, 'vllm'), 'streaming input text is only implemented for CosyVoice2 and do not support vllm!'  
                for i in self.llm.inference_bistream(text=text,  
                                                     prompt_text=prompt_text.to(self.device),  
                                                     prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32).to(self.device),  
                                                     prompt_speech_token=llm_prompt_speech_token.to(self.device),  
                                                     prompt_speech_token_len=torch.tensor([llm_prompt_speech_token.shape[1]], dtype=torch.int32).to(self.device),  
                                                     embedding=llm_embedding.to(self.device)):  
                    self.tts_speech_token_dict[uuid].append(i)  
            else:  
                for i in self.llm.inference(text=text.to(self.device),  
                                            text_len=torch.tensor([text.shape[1]], dtype=torch.int32).to(self.device),  
                                            prompt_text=prompt_text.to(self.device),  
                                            prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32).to(self.device),  
                                            prompt_speech_token=llm_prompt_speech_token.to(self.device),  
                                            prompt_speech_token_len=torch.tensor([llm_prompt_speech_token.shape[1]], dtype=torch.int32).to(self.device),  
                                            embedding=llm_embedding.to(self.device),  
                                            uuid=uuid):  
                    self.tts_speech_token_dict[uuid].append(i)  
        self.llm_end_dict[uuid] = True  
  
    def vc_job(self, source_speech_token, uuid):  
        self.tts_speech_token_dict[uuid] = source_speech_token.flatten().tolist()  
        self.llm_end_dict[uuid] = True  
  
    def token2wav(self, token, prompt_token, prompt_feat, embedding, uuid, finalize=False, speed=1.0):  
        with torch.cuda.amp.autocast(self.fp16):  
            tts_mel, self.flow_cache_dict[uuid] = self.flow.inference(token=token.to(self.device),  
                                                                      token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),  
                                                                      prompt_token=prompt_token.to(self.device),  
                                                                      prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),  
                                                                      prompt_feat=prompt_feat.to(self.device),  
                                                                      prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),  
                                                                      embedding=embedding.to(self.device),  
                                                                      flow_cache=self.flow_cache_dict[uuid])  
  
        # mel overlap fade in out  
        if self.mel_overlap_dict[uuid].shape[2] != 0:  
            tts_mel = fade_in_out(tts_mel, self.mel_overlap_dict[uuid], self.mel_window)  
        # append hift cache  
        if self.hift_cache_dict[uuid] is not None:  
            hift_cache_mel, hift_cache_source = self.hift_cache_dict[uuid]['mel'], self.hift_cache_dict[uuid]['source']  
            tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)  
        else:  
            hift_cache_source = torch.zeros(1, 1, 0)  
        # keep overlap mel and hift cache  
        if finalize is False:  
            self.mel_overlap_dict[uuid] = tts_mel[:, :, -self.mel_overlap_len:]  
            tts_mel = tts_mel[:, :, :-self.mel_overlap_len]  
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)  
            if self.hift_cache_dict[uuid] is not None:  
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)  
            self.hift_cache_dict[uuid] = {'mel': tts_mel[:, :, -self.mel_cache_len:],  
                                          'source': tts_source[:, :, -self.source_cache_len:],  
                                          'speech': tts_speech[:, -self.source_cache_len:]}  
            tts_speech = tts_speech[:, :-self.source_cache_len]  
        else:  
            if speed != 1.0:  
                assert self.hift_cache_dict[uuid] is None, 'speed change only support non-stream inference mode'  
```
