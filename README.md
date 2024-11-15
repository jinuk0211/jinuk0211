연락처
---------------
- https://www.threads.net/@rien_n_est
- 📫 dnr9333@gmail.com



기술스택 
----------------
#<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=#PyTorch&logoColor=D1180B"/>
<img src="https://img.shields.io/badge/C++-345F53?style=flat-square&logo=C++&logoColor=345F53"/>
<img
src="https://img.shields.io/badge/FASTAPI-43EED6?style=flat-square&logo=#009688&logoColor=43EED6"/>
<img src="https://img.shields.io/badge/Django-0A3711?style=flat-square&logo=#Django&logoColor=0A3711"/>
<img src="https://img.shields.io/badge/Flutter-99CCFF?style=flat-square&logo=#Flutter&logoColor=99CCFF"/>
<img src="https://img.shields.io/badge/Docker-00q9F4?style=flat-square&logo=#Docker&logoColor=0019F4"/>









대외활동
---------------------

dacon 고객 대출등급 분류 경진대회 - 상위 15%

dacon 직쏘퍼즐 ai 경진대회 - 상위 10%

cpp 코드 유사성 판단 ai 경진대회 - 상위 8%

캐글 LLM science exam -bronze medal 

캐글 LLM - Detect AI Generated Text -bronze medal

토익 960,

정보처리산업기사 

azure microsoft AI a1-900

및 aws certified cloud parctitioner 자격증 

Improved generative MCTS verifier with self critique 논문 1저자
[COLING2025] 

enhancing visual understanding capability of multimodal model through inference scaling 제1저자 - 대한전자공학회

SSM 모델 활용한 ecg데이터 예측 제1저자 - 정보통신학회 춘계우수논문 선정


AI 논문 리뷰 
-------------
<https://www.threads.net/@rien_n_est>



![jinuk's GitHub stats](https://github-readme-stats.vercel.app/api?username=jinuk0211&show_icons=true&theme=radical)
![junuk's GitHub stats](https://github-readme-stats.vercel.app/api?username=jinuk0211&show_icons=true&theme=radical)





attention is all we need
-----------
```python
class Attention(nn.Module):
    """attention is all you need"""

    def __init__(self, config, block_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.d_model
        self.num_heads = config.n_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_seq_len
        self.block_idx = block_idx
        if block_idx is None:
            logger.warning_once(
                 block_idx를 제공하지 않고 {self.class.name}를 인스턴스화하는 것은 권장되지 않으며,
                 caching을 사용할 경우 forward과정에서 문제가 발생할 수 있습니다.
                 attention class 생성시 block id 를 꼭 입력해주세요.
            )

        attn_config = config.attn_config
        self.attn_pdrop = attn_config.attn_pdrop
        self.clip_qkv = attn_config.clip_qkv
        self.num_key_value_heads = attn_config.kv_n_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.rope_theta = attn_config.rope_theta
        self.is_causal = True

        self.Wqkv = nn.Linear(
            self.hidden_size, self.hidden_size + 2 * self.num_key_value_heads * self.head_dim, bias=False
        )
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        bsz, q_len, _ = hidden_states.size() #batchsize

        qkv_states = self.Wqkv(hidden_states)
        min_val = -self.clip_qkv if self.clip_qkv is not None else None
        max_val = self.clip_qkv
        qkv_states = qkv_states.clamp(min=min_val, max=max_val)

        query_states, key_states, value_states = qkv_states.split(
            [
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                self.num_key_value_heads * self.head_dim,
            ],
            dim=2,
        )

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.block_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  #length에 관계없이 
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # fp32로 upcast
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attn_pdrop, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` 의 size는 {(bsz, self.num_heads, q_len, self.head_dim)}여야합니다 , 하지만 지금 size :"
                + f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.out_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

```
