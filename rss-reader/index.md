# RSS 使用说明

<div class="rss-page">
  <div class="rss-page__hero">
    <p class="rss-page__eyebrow">RSS WORKFLOW</p>
    <h1>先看说明，再进入阅读器</h1>
    <p>你可以在网页端添加订阅源、导入 OPML、导出 JSON，然后进入聚合阅读器统一刷新所有源。</p>
    <p><a href="./reader" class="rss-page__cta">打开 RSS 阅读器</a></p>
  </div>
</div>

## 你现在可用的能力

1. 手动添加 RSS 源（保存到当前浏览器 LocalStorage）。
2. 粘贴或上传 OPML 批量导入。
3. 导出当前订阅为 `rss-feeds.export.json`，用于回写仓库配置。
4. 阅读器页一键刷新全部订阅源，合并展示所有文章。

## 推荐使用流程

1. 先到阅读器页导入 OPML，并检查文章是否能正常拉取。  
2. 确认订阅列表无误后点击 `导出 JSON`。  
3. 用导出的内容覆盖 `.vitepress/rss-feeds.json` 并提交仓库。  
4. 这样你所有设备打开站点时都能加载同一套默认订阅源。

## GitHub Pages 注意事项

- GitHub Pages 是静态托管，网页端新增不会自动写回仓库文件。
- LocalStorage 仅在当前浏览器生效，换设备不会同步。
- 某些源可能被 CORS 限制，必要时给该源设置代理前缀。

::: tip
阅读器页地址：`/rss-reader/reader`
:::
