# Markdown语法完全指南

## 1. 标题语法
要创建标题，请在单词或短语前面添加井号 (`#`) 。`#` 的数量代表了标题的级别。
例如，添加三个 `#` 表示创建一个三级标题 (`<h3>`) (例如：`### My Header`)。

|Markdown语法|HTML|预览效果|
|---|---|---|
|`# Heading level 1`|`<h1>Heading level 1</h1>`|<h1>Heading level 1</h1>|
|`## Heading level 2`|`<h2>Heading level 2</h2>`|<h2>Heading level 2</h2>|
|`### Heading level 3`|`<h3>Heading level 3</h3>`|<h3>Heading level 3</h3>|
|`#### Heading level 4`|`<h4>Heading level 4</h4>`|<h4>Heading level 4</h4>|
|`##### Heading level 5`|`<h5>Heading level 5</h5>`|<h5>Heading level 5</h5>|
|`###### Heading level 6`|`<h6>Heading level 6</h6>`|<h6>Heading level 6</h6>|
::: tip 提示
**实战建议**：一般markdown文档中建议最多使用到4级标题，避免层级混乱。
:::
**可选语法**
还可以在文本下方添加任意数量的 == 号来标识一级标题，或者 -- 号来标识二级标题。

| Markdown语法 | HTML | 预览效果 |
|--------------|------|----------|
| <code>Heading level 1<br>===============</code> | `<h1>Heading level 1</h1>` | <h1 style="font-size:2em; color:#333; margin:0">Heading level 1</h1> |
| <code>Heading level 2<br>---------------</code> | `<h2>Heading level 2</h2>` | <h2 style="font-size:1.5em; color:#333; margin:0">Heading level 2</h2> |

::: info
不同的 Markdown 应用程序处理 `#` 和标题之间的空格方式并不一致。为了兼容考虑，请用一个空格在 `#` 和标题之间进行分隔。
:::
## 2. 段落语法 
要创建段落，请使用空白行将一行或多行文本进行分隔。并且不要用空格（spaces）或制表符（ tabs）缩进段落。
| Markdown语法 | HTML | 预览效果 |
|---|---|---|
| `I really like using Markdown.`<br><br>`I think I'll use it to format all of my documents from now on.` | `<p>I really like using Markdown.</p>`<br><br>`<p>I think I'll use it to format all of my documents from now on.</p>` | I really like using Markdown.<br><br>I think I'll use it to format all of my documents from now on. |

## 3. 换行语法
在一行的末尾添加两个或多个空格，然后按回车键,即可创建一个换行(`<br>`)。  
| Markdown语法 | HTML | 预览效果 |
|---|---|---|
| `This is the first line.  `<br>`And this is the second line.` | `<p>This is the first line.<br>`<br>`And this is the second line.</p>` | This is the first line.<br>And this is the second line. |

几乎每个 Markdown 应用程序都支持两个或多个空格进行换行，称为 `结尾空格（trailing whitespace) `的方式，但这是有争议的，因为很难在编辑器中直接看到空格，并且很多人在每个句子后面都会有意或无意地添加两个空格。由于这个原因，你可能要使用除结尾空格以外的其它方式来换行。幸运的是，几乎每个 Markdown 应用程序都支持另一种换行方式：HTML 的 `<br>`标签。  <br>
为了兼容性，请在行尾添加“结尾空格”或 HTML 的 `<br>` 标签来实现换行。  


## 4. 强调语法  
**粗体**  
要加粗文本，请在单词或短语的前后各添加两个星号（asterisks）或下划线（underscores）。如需加粗一个单词或短语的中间部分用以表示强调的话，请在要加粗部分的两侧各添加两个星号（asterisks）。  
**斜体**  
要用斜体显示文本，请在单词或短语前后添加一个星号（asterisk）或下划线（underscore）。要斜体突出单词的中间部分，请在字母前后各添加一个星号，中间不要带空格。  
**粗体和斜体**  
显示文本，请在单词或短语的前后各添加三个星号或下划线。要加粗并用斜体显示单词或短语的中间部分，请在要突出显示的部分前后各添加三个星号，中间不要带空格。
**删除线**
您可以通过在单词中心放置一条水平线来删除单词。结果看起来像这样。此功能使您可以指示某些单词是一个错误，要从文档中删除。若要删除单词，请在单词前后使用两个波浪号`~~`。


|效果|语法|适用场景|示例|
|---|---|---|---|
|**加粗**|`**文本**`|核心概念、函数名|**nn.Linear()** 是PyTorch线性层|
|*斜体*|`*文本*`|补充说明、变量名|设 *x* 为输入张量|
|***加粗斜体***|`***文本***`|核心结论、重要提醒|***反向传播是深度学习的核心***|
|~~删除线~~|`~~文本~~`|废弃知识点、错误示例|~~Python中使用var关键字声明变量~~|
|`行内代码`|\``代码片段`\`|函数、命令、变量|使用 `np.array()` 创建数组|  

## 5. 引用语法

**1.** 要创建块引用，请在段落前添加一个 `>` 符号。

`> Dorothy followed her through many of the beautiful rooms in her castle.`  <br>

渲染效果如下所示：

> Dorothy followed her through many of the beautiful rooms in her castle.

**2.** 多个段落的块引用

块引用可以包含多个段落。为段落之间的空白行添加一个 `>` 符号。
```markdown
> Dorothy followed her through many of the beautiful rooms in her castle.
>
> The Witch bade her clean the pots and kettles and sweep the floor and keep the fire fed with wood.
```  

渲染效果如下：

> Dorothy followed her through many of the beautiful rooms in her castle.
>
> The Witch bade her clean the pots and kettles and sweep the floor and keep the fire fed with wood.

**3.** 嵌套块引用  

块引用可以嵌套。在要嵌套的段落前添加一个 `>>` 符号。
```markdown
> Dorothy followed her through many of the beautiful rooms in her castle.
>
>> The Witch bade her clean the pots and kettles and sweep the floor and keep the fire fed with wood.
``` 

渲染效果如下：

> Dorothy followed her through many of the beautiful rooms in her castle.
>
>> The Witch bade her clean the pots and kettles and sweep the floor and keep the fire fed with wood.

**4.** 带有其它元素的块引用<br>

块引用可以包含其他 Markdown 格式的元素。并非所有元素都可以使用，你需要进行实验以查看哪些元素有效。
```markdown
> **The quarterly results look great!**
>
> - Revenue was off the chart.
> - Profits were higher than ever.
>
>  *Everything* is going according to **plan**.
```
渲染效果如下：

> **The quarterly results look great!**
>
> - Revenue was off the chart.
> - Profits were higher than ever.
>
>  *Everything* is going according to **plan**.

## 6. 列表语法

可以将多个条目组织成有序或无序列表。

### 有序列表
要创建有序列表，请在每个列表项前添加数字并紧跟一个英文句点。数字不必按数学顺序排列，但是列表应当以数字 1 起始。

<div style="text-align: center">

| Markdown 语法 | HTML | 预览效果 |
| :--- | :--- | :--- |
| `1.First item`<br>`2.Second item`<br>`3.Third item`<br>`4.Fourth item` | `<ol>`<br>`  <li>First item</li>`<br>  `<li>Second item</li>`<br>  `<li>Third item</li>`<br>  `<li>Fourth item</li>`<br>`</ol>` | <ol><li>First item</li><li>Second item</li><li>Third item</li><li>Fourth item</li></ol> |
| `1.First item`<br>`1.Second item`<br>`1.Third item`<br>`1.Fourth item` |`<ol>`<br>`  <li>First item</li>`<br>  `<li>Second item</li>`<br>  `<li>Third item</li>`<br>  `<li>Fourth item</li>`<br>`</ol>` | <ol><li>First item</li><li>Second item</li><li>Third item</li><li>Fourth item</li></ol> |
| `1.First item`<br>`8.Second item`<br>`3.Third item`<br>`5.Fourth item` | `<ol>`<br>`  <li>First item</li>`<br>  `<li>Second item</li>`<br>  `<li>Third item</li>`<br>  `<li>Fourth item</li>`<br>`</ol>` | <ol><li>First item</li><li>Second item</li><li>Third item</li><li>Fourth item</li></ol> |
|<code>1.First item</code><br><code>2.Second item</code><br><code>3.Third item</code><br><code>&nbsp;&nbsp;&nbsp;&nbsp;1.Indented item</code><br><code>&nbsp;&nbsp;&nbsp;&nbsp;2.Indented item</code><br><code>4.Fourth item</code>| `<ol>`<br>  `<li>First item</li>`<br>  `<li>Second item</li>`<br>  `<li>Third item`<br>    `<ol>`<br>      `<li>Indented item</li>`<br>      `<li>Indented item</li>`<br>    `</ol>`<br>  `</li>`<br>  `<li>Fourth item</li>`<br>`</ol>` | <ol><li>First item</li><li>Second item</li><li>Third item<ol><li>Indented item</li><li>Indented item</li></ol></li><li>Fourth item</li></ol> |

</div>


### 无序列表  
要创建无序列表，请在每个列表项前面添加破折号 (`-`)、星号 (`*`) 或加号 (`+`) 。缩进一个或多个列表项可创建嵌套列表。
<div style="text-align: center">

| Markdown 语法 | HTML | 预览效果 |
| :--- | :--- | :--- |
| `- First item`<br>`- Second item`<br>`- Third item`<br>`- Fourth item` | `<ul>`<br>`  <li>First item</li>`<br>  `<li>Second item</li>`<br>  `<li>Third item</li>`<br>  `<li>Fourth item</li>`<br>`</ul>` | <ul><li>First item</li><li>Second item</li><li>Third item</li><li>Fourth item</li></ul> |
| `* First item`<br>`* Second item`<br>`* Third item`<br>`* Fourth item` |`<ul>`<br>`  <li>First item</li>`<br>  `<li>Second item</li>`<br>  `<li>Third item</li>`<br>  `<li>Fourth item</li>`<br>`</ul>` | <ul><li>First item</li><li>Second item</li><li>Third item</li><li>Fourth item</li></ul> |
| `+ First item`<br>`+ Second item`<br>`+ Third item`<br>`+ Fourth item` | `<ul>`<br>`  <li>First item</li>`<br>  `<li>Second item</li>`<br>  `<li>Third item</li>`<br>  `<li>Fourth item</li>`<br>`</ul>` | <ul><li>First item</li><li>Second item</li><li>Third item</li><li>Fourth item</li></ul> |
|<code>- First item</code><br><code>- Second item</code><br><code>- Third item</code><br><code>&nbsp;&nbsp;&nbsp;&nbsp;- Indented item</code><br><code>&nbsp;&nbsp;&nbsp;&nbsp;- Indented item</code><br><code>- Fourth item</code>| `<ul>`<br>  `<li>First item</li>`<br>  `<li>Second item</li>`<br>  `<li>Third item`<br>    `<ul>`<br>      `<li>Indented item</li>`<br>      `<li>Indented item</li>`<br>    `</ul>`<br>  `</li>`<br>  `<li>Fourth item</li>`<br>`</ul>` | <ul><li>First item</li><li>Second item</li><li>Third item<ul><li>Indented item</li><li>Indented item</li></ul></li><li>Fourth item</li></ul> |

</div>


### 列表嵌套

在列表中嵌套其他元素
要在保留列表连续性的同时在列表中添加另一种元素，请将该元素缩进四个空格或一个制表符，如下例所示：

段落
```markdown
*   This is the first list item.
*   Here's the second list item.

    I need to add another paragraph below the second list item.

*   And here's the third list item.
```
渲染效果如下：

*   This is the first list item.
*   Here's the second list item.

    I need to add another paragraph below the second list item.

*   And here's the third list item.

**引用块**
```markdown
*   This is the first list item.
*   Here's the second list item.

    > A blockquote would look great below the second list item.

*   And here's the third list item.
```
渲染效果如下：

*   This is the first list item.
*   Here's the second list item.

    > A blockquote would look great below the second list item.

*   And here's the third list item.

**代码块**  
代码块通常采用四个空格或一个制表符缩进。当它们被放在列表中时，请将它们缩进八个空格或两个制表符。
```markdown
1.  Open the file.
2.  Find the following code block on line 21:

        &lt;html>
          &lt;head>
            &lt;title>Test&lt;/title>
          &lt;/head>

3.  Update the title to match the name of your website.
```
渲染效果如下：

1.  Open the file.
2.  Find the following code block on line 21:
```html
        &lt;html>
          &lt;head>
            &lt;title>Test&lt;/title>
          &lt;/head>
```
3.  Update the title to match the name of your website.

**图片**
```markdown
1.  Open the file containing the Linux mascot.
2.  Marvel at its beauty.

    ![Tux, the Linux mascot](./images/linux_tux.png)

3.  Close the file.
```
渲染效果如下：

1.  Open the file containing the Linux mascot.
2.  Marvel at its beauty.

    ![Tux, the Linux mascot](./images/linux_tux.png)

3.  Close the file.

**列表**
```markdown
You can nest an unordered list in an ordered list, or vice versa.

1. First item
2. Second item
3. Third item
    - Indented item
    - Indented item
4. Fourth item
```
渲染效果如下：

You can nest an unordered list in an ordered list, or vice versa.

1. First item
2. Second item
3. Third item
    - Indented item
    - Indented item
4. Fourth item

## 7. 代码块语法

要将单词或短语表示为代码，请将其包裹在反引号 (`` ` ``) 中。代码块之前和之后的行上使用三个反引号（ ```` ``` ````）或三个波浪号（`~~~`）
| Markdown 语法 | HTML | 预览效果 |
| :--- | :--- | :--- |
|``At the command prompt, type `nano`.``	|`At the command prompt, type <code>nano</code>.`	|At the command prompt, type `nano`.|

**转义反引号**
如果你要表示为代码的单词或短语中包含一个或多个反引号，则可以通过将单词或短语包裹在双反引号(``` `` ```)中。

| Markdown 语法 | HTML | 预览效果 |
| :--- | :--- | :--- |
|``` ``Use `code` in your Markdown file.`` ```	|`` <code>Use `code` in your Markdown file.</code> ``	|Use `code` in your Markdown file.|

### 进阶用法（行号/高亮）

VitePress内置支持，通过`{行号范围}`高亮关键代码，`:line-numbers`显示行号,支持多种编程语言高亮，语法名需准确（如`python`而非`Python`）, `// [!code focus]`进行聚焦高亮,`// [!code ++]`新增,`// [!code --]`减少，其余也可聚焦`// [!code error]`，`// [!code warning]`。

``````markdown
```python:line-numbers {2,5-7}
# 高亮第2行和5-7行代码
def train_step(model, x, y, optimizer, criterion):
    # 前向传播
    pred = model(x)
    loss = criterion(pred, y)  # 计算损失
    # 反向传播
    loss.backward()            # 梯度计算
    optimizer.step()           # 参数更新
    optimizer.zero_grad()      # 清空梯度  
    return loss.item()
```
``````

语法高亮结果：
```python:line-numbers {2,5-7} 
# 高亮第2行和5-7行代码
def train_step(model, x, y, optimizer, criterion):
    # 前向传播
    pred = model(x) 
    loss = criterion(pred, y)  # 计算损失
    # 反向传播
    loss.backward()            # 梯度计算
    optimizer.step()           # 参数更新
    optimizer.zero_grad()      # 清空梯度 
    return loss.item()
```
聚焦结果：
```python:line-numbers
print("Hello, World!")  # 新增 // [!code ++]
print("Hello, World!")  # hello // [!code focus]
print("Hello, World!")  # 减少 // [!code --]
```

### 代码块导入功能
根据给定地址导入代码片段,使用绝对路径时需要加上`@`符号,可以指定语言
```markdown
<<< ./code/import_code.py {python} # 根据给定地址导入代码片段
```
<<< ./code/import_code.py {python}

也可以导入markdown文件
```markdown
<!--@include: ./code/import_code.md-->
```
导入结果如下:  
<!--@include: ./code/import_md.md-->
### 代码分组功能
可以将多个代码块分组，方便切换
``````markdown
::: code-group

```js [config.js]
/**
 * @type {import('vitepress').UserConfig}
 */
const config = {
  // ...
}

export default config
```

```ts [config.ts]
import type { UserConfig } from 'vitepress'

const config: UserConfig = {
  // ...
}

export default config
```
:::

``````

::: code-group

```js [config.js]
/**
 * @type {import('vitepress').UserConfig}
 */
const config = {
  // ...
}

export default config
```

```ts [config.ts]
import type { UserConfig } from 'vitepress'

const config: UserConfig = {
  // ...
}

export default config
```
:::


## 8. 分隔线语法

要创建分隔线，请在单独一行上使用三个或多个星号 (`***`)、破折号 (`---`) 或下划线 (`___`) ，并且不能包含其他内容。
```markdown
***

---

_________________
```
以上三个分隔线的渲染效果看起来都一样：

***

## 9. 链接语法

链接文本放在中括号内，链接地址放在后面的括号中，链接title可选。

超链接Markdown语法代码：`[超链接显示名](超链接地址 "超链接title")`

对应的HTML代码：`<a href="超链接地址" title="超链接title">超链接显示名</a>`

```markdown
这是一个链接 [Markdown语法](https://markdown.com.cn)。
```

渲染效果如下：

这是一个链接 [Markdown语法](https://markdown.com.cn)。

**给链接增加 Title**

链接title是当鼠标悬停在链接上时会出现的文字，这个title是可选的，它放在圆括号中链接地址后面，跟链接地址之间以空格分隔。  
外部链接必须带 `https://`/`http://`，否则会被识别为内部路径。
```markdown
这是一个链接 [Markdown语法](https://markdown.com.cn "最好的markdown教程")。
```
渲染效果如下：

这是一个链接 [Markdown语法](https://markdown.com.cn "最好的markdown教程")。

**网址和Email地址**
使用尖括号可以很方便地把URL或者email地址变成可点击的链接。
```markdown
<https://markdown.com.cn>
<fake@example.com>
```
渲染效果如下：  
<https://markdown.com.cn>
<fake@example.com>

**带格式化的链接**
强调 链接, 在链接语法前后增加星号。 要将链接表示为代码，请在方括号中添加反引号。
```markdown
I love supporting the **[EFF](https://eff.org)**.
This is the *[Markdown Guide](https://www.markdownguide.org)*.
See the section on [`code`](#code).
```
渲染效果如下：  
I love supporting the **[EFF](https://eff.org)**.  
This is the *[Markdown Guide](https://www.markdownguide.org)*.  
See the section on [`code`](#code).

## 10. 图片语法

要添加图像，请使用感叹号 (`!`), 然后在方括号增加替代文本，图片链接放在圆括号里，括号里的链接后可以增加一个可选的图片标题文本。

插入图片Markdown语法代码：`![图片alt](图片链接 "图片title")`。

对应的HTML代码：`<img src="图片链接" alt="图片alt" title="图片title">`

```markdown
![这是图片](./images/magic-garden.jpg "Magic Gardens")
```
渲染效果如下：
![这是图片](./images/magic-garden.jpg "Magic Gardens")

**链接图片**
给图片增加链接，请将图像的Markdown 括在方括号中，然后将链接添加在圆括号中。

```markdown
[![沙漠中的岩石图片](./images/shiprock.jpg "Shiprock")](https://markdown.com.cn)
```
渲染效果如下：  
[![沙漠中的岩石图片](./images/shiprock.jpg "Shiprock")](https://markdown.com.cn)

## 11.转义字符语法
要显示原本用于格式化 Markdown 文档的字符，请在字符前面添加反斜杠字符 `\` 。
```markdown
\* Without the backslash, this would be a bullet in an unordered list.
```
渲染效果如下：  
\* Without the backslash, this would be a bullet in an unordered list.

### 可做转义的字符
|字符|名称|
|---|---|
|`\`|反斜杠|
|`` ` ``|反引号|
|`*`|星号|
|`_`|下划线|
|`{ }`|花括号|
|`[ ]`|方括号|
|`( )`|圆括号|
|`#`|井号|
|`+`|加号|
|`-`|减号|
|`.`|点|
|`!`|感叹号|
|`\|`|竖线|

**特殊字符自动转义**
在 HTML 文件中，有两个字符需要特殊处理： `<` 和 `&` 。 < 符号用于起始标签，`&` 符号则用于标记 HTML 实体，如果你只是想要使用这些符号，你必须要使用实体的形式，像是 `&lt;` 和 `&amp;`。

`&` 符号其实很容易让写作网页文件的人感到困扰，如果你要打`「AT&T」` ，你必须要写成`「AT&amp;T」` ，还得转换网址内的 `&` 符号，如果要链接到：
```markdown
http://images.google.com/images?num=30&q=larry+bird
```
必须要把网址转成：
```markdown
http://images.google.com/images?num=30&amp;q=larry+bird
```

## 12. 内嵌 HTML 标签语法

对于 Markdown 涵盖范围之外的标签，都可以直接在文件里面用 HTML 本身。如需使用 HTML，不需要额外标注这是 HTML 或是 Markdown，只需 HTML 标签添加到 Markdown 文本中即可。

**行级內联标签**

HTML 的行级內联标签如 `<span>`、`<cite>`、`<del>` 不受限制，可以在 Markdown 的段落、列表或是标题里任意使用。依照个人习惯，甚至可以不用 Markdown 格式，而采用 HTML 标签来格式化。例如：如果比较喜欢 HTML 的 `<a>` 或 `<img>` 标签，可以直接使用这些标签，而不用 Markdown 提供的链接或是图片语法。当你需要更改元素的属性时（例如为文本指定颜色或更改图像的宽度），使用 HTML 标签更方便些。

HTML 行级內联标签和区块标签不同，在內联标签的范围内， Markdown 的语法是可以解析的。

```markdown
This **word** is bold. This <em>word</em> is italic.  
```

渲染效果如下:

This **word** is bold. This <em>word</em> is italic.

**区块标签**

区块元素──比如 `<div>`、`<table>`、`<pre>`、`<p>` 等标签，必须在前后加上空行，以便于内容区分。而且这些元素的开始与结尾标签，不可以用 tab 或是空白来缩进。Markdown 会自动识别这区块元素，避免在区块标签前后加上没有必要的 `<p>` 标签。

例如，在 Markdown 文件里加上一段 HTML 表格：
```markdown
This is a regular paragraph.

<table>
    <tr>
        <td>Foo</td>
    </tr>
</table>

This is another regular paragraph.
```

请注意，Markdown 语法在 HTML 区块标签中将不会被进行处理。无法在 HTML 区块内使用 Markdown 形式的`*强调*`。

## 二. 避坑指南与效率技巧

### 1. 常见问题解决

**代码块无高亮？** 检查编程语言名是否正确（如`cpp`而非`c++`，`bash`而非`shell`）。

**数学公式不渲染？** 确认已安装`markdown-it-mathjax3`插件，且配置中`markdown.math: true`已开启。

**图片无法显示？** 检查图片路径是否正确，建议统一放在`public/images`目录，引用路径以`/images/`开头。

**链接跳转失效？** 优先使用相对路径，锚点链接需与标题完全匹配（区分大小写）。

### 2. 高效编辑快捷键（VS Code）

|操作目标|Markdown语法|VS Code快捷键|
|---|---|---|
|加粗文本|`**文本**`|Ctrl + B|
|斜体文本|`*文本*`|Ctrl + I|
|插入链接|`[文本](链接)`|Ctrl + K + L|
|插入代码块|````代码````|Ctrl + K + F|
|插入图片|`![描述](路径)`|Ctrl + Shift + I|
|创建列表|`- 内容`/`1. 内容`|输入`-`/`1.` + 空格|
### 3. 实用插件推荐（VS Code）

- **Markdown All in One**：提供快捷键、目录生成、预览等一站式功能

- **Markdown Preview Enhanced**：实时预览Markdown效果，支持数学公式渲染

- **Paste Image**：截图后直接粘贴为图片，自动保存到指定目录

- **LaTeX Workshop**：增强数学公式编辑体验，支持代码补全

## 三. 语法速查表

|功能分类|常用语法|备注|
|---|---|---|
|标题|`# 一级标题`|对应页面大标题|
||`## 二级标题`|对应侧边栏一级菜单|
|文本|`**加粗**`|强调核心概念|
||``行内代码``|函数、变量等|
||`> 引用文本`|文献、注释|
|代码|````python 代码 ````|指定语言高亮|
||`{行号} showLineNumbers`|高亮+行号|
|公式|`$行内公式$`|嵌入正文|
||`$$块级公式$$`|独立显示|
|链接|`[文本](链接)`|外部/内部链接|
||`[文本](#锚点)`|页面内跳转|
|图片|`<img src="路径" width="尺寸">`|控制显示大小|

其余Markdown语法可查看[Vitepress官方文档](https://vitepress.dev/zh/guide/markdown)和[Markdown教程](https://markdown.com.cn/basic-syntax)
