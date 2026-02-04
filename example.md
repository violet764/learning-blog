<div style="
  background-color: #f8f9fa;
  padding: 20px 24px;
  border-radius: 12px;
  border-left: 5px solid #42b983;
  margin: 20px 0;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
">
  <div style="
    color: #42b983;
    font-weight: 600;
    font-size: 1.1em;
    margin-bottom: 12px;
  ">
    🔥 ZeroTIR Paper, 🐍 openrlhf_async_pipeline
  </div>
  <div style="color: #333; line-height: 1.6;">
    单工具场景，提出<mark style="background: linear-gradient(120deg, #a1c4fd 0%, #c2e9fb 100%); padding: 2px 6px; border-radius: 4px;">ZeroTIR</mark>，仅靠任务<mark style="background: linear-gradient(120deg, #ffecd2 0%, #fcb69f 100%); padding: 2px 6px; border-radius: 4px;">Reward</mark>，让模型自主编写python解决数学问题😊。
  </div>
</div>



## Agent RL Scaling Law

<div class="card">
  <div class="card-content">
    <div class="card-title">🔥 ZeroTIR Paper, 🐍 openrlhf_async_pipeline</div>
    <div class="card-body">
      单工具场景，提出ZeroTIR，仅靠任务Reward，让模型自主编写python解决数学问题😊。
    </div>
    <div class="card-footer">
      做了一些scaling的实验，模型越大越好、交互代码次数并非越多越好😉。
    </div>
  </div>
</div>

--- 

### 算法名称：线性回归


<div class="animated-card" style="
  background-color: #fff5f5;
  padding: 20px 24px;
  border-radius: 12px;
  border-left: 5px solid #f56565;
  margin: 20px 0;
  box-shadow: 0 4px 6px rgba(245, 101, 101, 0.1);
">
  <div style="color: #4a5568; line-height: 1.7;">
    <p style="margin: 8px 0;">
      当 $x$ 增加时，$y$ 的变化率是 $w$
    </p>
    
    <p style="margin: 8px 0;">
      公式：$y = w_1x_1 + w_2x_2 + b$
    </p>
    
    <p style="margin: 8px 0;">
      损失函数：$L = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$
    </p>
  </div>
</div>


<div class="animated-card" style="
  background-color: #f8fafc;
  padding: 20px 24px;
  border-radius: 12px;
  border-left: 5px solid #4299e1;
  margin: 20px 0;
  box-shadow: 0 4px 6px rgba(66, 153, 225, 0.1);
">
  <h4 style="margin: 0 0 16px 0; color: #2d3748; font-weight: 600;">
    算法名称：线性回归
  </h4>
  
  <div style="color: #4a5568; line-height: 1.7;">
    <p style="margin: 8px 0;">
      <strong class="subtle-highlight">适用场景：</strong>连续值预测（房价、销量）
    </p>
    
    <!-- 数学公式要这样写 -->
    <p style="margin: 12px 0; text-align: center;">
      $$ y = wx + b $$
    </p>
    
    <p style="margin: 8px 0;">
      <strong class="subtle-highlight">优缺点：</strong>简单易解释，但无法拟合非线性关系
    </p>
  </div>
</div>