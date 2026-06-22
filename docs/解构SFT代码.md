## 模型loss及其定义

MiniOneRec 的 SFT 没有自定义 loss，而是使用 Qwen2 因果语言模型内置的 token-level Cross Entropy Loss（交叉熵损失）。

调用链如下：

```text
transformers.Trainer
  → Qwen2ForCausalLM.forward(labels=...)
  → ForCausalLMLoss
  → torch.nn.functional.cross_entropy
```

### Loss 在哪里定义

| 环节 | 位置 | 作用 |
| --- | --- | --- |
| 创建 Trainer | `sft_mps.py` 中的 `transformers.Trainer(...)` | 将 model、dataset 和 data collator 交给 Hugging Face Trainer |
| 构造 labels | `data.py` 的各个 SFT Dataset | 指定哪些 token 参与 loss |
| Qwen 计算 loss | `transformers/models/qwen2/modeling_qwen2.py` | 模型收到 labels 后调用 `self.loss_function(...)` |
| Causal LM loss | `transformers/loss/loss_utils.py` 的 `ForCausalLMLoss` | shift labels 并调用 `cross_entropy` |

`sft_mps.py` 没有传入自定义 `compute_loss`。因此 Trainer 调用模型时，只要 batch 中包含 `labels`，`Qwen2ForCausalLM.forward()` 就会返回模型内部计算的 `loss`。

### Labels 如何构造

数据集将 prompt 和目标回答拼接为一条 token 序列：

```python
golden_tokens = tokenizer.encode(target, bos=False, eos=True)
input_prompt_len = len(tokens)
tokens = tokens + golden_tokens
labels = [-100] * input_prompt_len + tokens[input_prompt_len:]
```

对应关系为：

```text
input_ids: [Prompt tokens                    ][Response tokens]
labels:    [-100, -100, ..., -100            ][Response token IDs]
```

`cross_entropy` 使用 `ignore_index=-100`，所以：

- Prompt 部分不计算 loss；
- Response 部分计算 loss；
- Response 末尾的 EOS token 也参与训练。

例如：

```text
Prompt：历史 SID → 请预测下一个商品
Response：<a_1><b_2><c_3>
```

loss 只监督模型生成 `<a_1>`、`<b_2>`、`<c_3>` 和 EOS，不要求模型复现 Prompt。

### Causal LM 的 shift

因果语言模型使用位置 `n` 的输出预测位置 `n+1` 的 token。Transformers 的核心处理等价于：

```python
labels = torch.nn.functional.pad(labels, (0, 1), value=-100)
shift_labels = labels[..., 1:]

loss = torch.nn.functional.cross_entropy(
    logits.view(-1, vocab_size),
    shift_labels.reshape(-1),
    ignore_index=-100,
)
```

因此，loss 衡量的是每个有效位置对“下一个目标 token”的预测误差，而不是整段文本级别的单一分类误差。

### 三个训练 Dataset 的 Loss 目标

三个 Dataset 使用相同的 Causal LM Cross Entropy Loss，但 Response 不同：

| Dataset | 输入 | Response/监督目标 |
| --- | --- | --- |
| `SidSFTDataset` | 历史 SID 序列 | 下一个 item 的 SID |
| `SidItemFeatDataset` | 商品标题或 SID | SID 或商品标题 |
| `FusionSeqRecDataset` | 历史 SID 序列 | 下一个 item 的自然语言标题 |

`ConcatDataset` 将这些任务组合成多任务训练集。Trainer 不会为它们选择不同的 loss；区别完全来自各 Dataset 构造的 `input_ids` 和 `labels`。

### `freeze_LLM=True` 对 Loss 的影响

`freeze_LLM=True` 不改变 loss 的定义和数值计算方式，只改变哪些参数接收并应用梯度：

```text
Transformer 主体          不更新
旧 token embedding        梯度被清零，不更新
新增 SID token embedding  更新
```

也就是说，模型仍然对所有 Response token 计算同一种交叉熵；但优化器最终只更新新增 SID token 对应的 embedding 行。

## batch_size 与 micro_batch_size

### 核心区别与计算关系

在当前单设备 MPS 实现中：

- `micro_batch_size`：一次 forward/backward 真正送入 GPU 的样本数，直接影响单次计算规模、GPU 并行度和内存占用；
- `batch_size`：执行一次 `optimizer.step()` 前累计参与梯度计算的总样本数，也称 effective batch size（有效批大小）。

`sft_mps.py` 的计算方式是：

```python
gradient_accumulation_steps = batch_size // micro_batch_size
```

因此单设备下：

```text
batch_size = micro_batch_size × gradient_accumulation_steps
```

当前代码要求 `batch_size >= micro_batch_size >= 1`，且 `batch_size` 必须能被 `micro_batch_size` 整除。

### 训练过程示例

当 `micro_batch_size=1, batch_size=8` 时，梯度累积次数为 8：

```text
样本 1 → forward/backward ┐
样本 2 → forward/backward │
...                       ├→ optimizer.step()
样本 8 → forward/backward ┘
```

GPU 每次只处理 1 条数据，需要执行 8 次 forward/backward 才更新一次参数。

当 `micro_batch_size=2, batch_size=8` 时，梯度累积次数为 4：

```text
样本 1～2 → forward/backward ┐
样本 3～4 → forward/backward │
样本 5～6 → forward/backward ├→ optimizer.step()
样本 7～8 → forward/backward ┘
```

两种配置的 effective batch size 都是 8，参数更新次数和每次更新使用的样本总数相同，但单次 GPU 负载、调度次数和内存占用不同。

### 为什么影响 GPU 利用率

GPU 擅长并行矩阵运算。`micro_batch_size=1` 时，单次计算规模较小，CPU 数据整理、Metal kernel 启动和 CPU/GPU 同步等固定开销占比更高，GPU 可能在两次任务提交之间空闲。

适当增大 `micro_batch_size` 通常可以：

- 增大单次矩阵计算规模，提高 GPU 并行度；
- 减少完成同一 effective batch 所需的 forward/backward 次数；
- 降低 kernel 调度和同步开销占比；
- 提高 samples/s 或 tokens/s。

代价是激活值和临时张量占用增加，更容易触发 MPS OOM。动态 Padding 还会将同一 micro batch 中的序列补到最长长度；序列长度差异较大时，增大 micro batch 可能增加无效计算。

| 配置 | 每次 GPU 处理样本 | 梯度累积次数 | Effective batch | 内存压力 |
| --- | ---: | ---: | ---: | --- |
| `micro=1, batch=8` | 1 | 8 | 8 | 最低 |
| `micro=2, batch=8` | 2 | 4 | 8 | 较低 |
| `micro=4, batch=8` | 4 | 2 | 8 | 较高 |
| `micro=8, batch=8` | 8 | 1 | 8 | 最高 |

只增大 `batch_size`、保持 `micro_batch_size=1`，只会增加梯度累积次数，不会增大一次送入 GPU 的工作量，因此通常不能改善 GPU 利用率。

### 排障与调参方法

在当前 24 GB Apple Silicon 上，可保持 `batch_size=8`，依次测试：

```text
micro_batch_size=1 → 2 → 4
```

每档保持数据、模型、`cutoff_len` 和训练步数一致，比较：

1. samples/s 或 tokens/s；
2. `powermetrics` 的 `GPU HW active residency`；
3. 单个 optimizer step 耗时；
4. macOS 内存压力及是否出现 MPS OOM；
5. loss 是否为有限值且变化正常。

| 现象 | 可能原因 | 调整方向 |
| --- | --- | --- |
| GPU 利用率低、内存余量大 | micro batch 太小或同步频繁 | 增大 `micro_batch_size` |
| GPU 利用率提高但吞吐未提高 | Padding 浪费、内存带宽或同步成为瓶颈 | 按长度分组或退回上一档 |
| MPS OOM 或内存压力变黄/红 | micro batch 过大 | 降低 `micro_batch_size` 或 `cutoff_len` |
| 增大 `batch_size` 后速度未提高 | 只增加了梯度累积次数 | 固定 effective batch，调整 `micro_batch_size` |

Hugging Face Trainer 的进度条通常按 optimizer step 计数。一个进度条 step 可能包含多次 micro batch forward/backward，因此分析 `秒/step` 时必须结合 `gradient_accumulation_steps`。
