const fs = require('fs');

// 读取订阅源文件
const feedsData = JSON.parse(fs.readFileSync('./.vitepress/rss-feeds.json', 'utf8'));

console.log('开始检测重复的订阅源...\n');

// 用于检测重复的URL和标题
const urlMap = new Map();
const titleMap = new Map();
const duplicates = [];

// 分析每个订阅源
feedsData.forEach((feed, index) => {
  const normalizedUrl = feed.url.toLowerCase().trim();
  const normalizedTitle = feed.title.toLowerCase().trim();
  
  // 检查URL重复
  if (urlMap.has(normalizedUrl)) {
    duplicates.push({
      type: 'URL重复',
      original: urlMap.get(normalizedUrl),
      duplicate: { index, title: feed.title, url: feed.url }
    });
  } else {
    urlMap.set(normalizedUrl, { index, title: feed.title });
  }
  
  // 检查标题重复
  if (titleMap.has(normalizedTitle)) {
    duplicates.push({
      type: '标题重复',
      original: titleMap.get(normalizedTitle),
      duplicate: { index, title: feed.title, url: feed.url }
    });
  } else {
    titleMap.set(normalizedTitle, { index, title: feed.title });
  }
});

// 输出结果
if (duplicates.length > 0) {
  console.log(`发现 ${duplicates.length} 个重复项：\n`);
  
  duplicates.forEach((dup, i) => {
    console.log(`${i + 1}. ${dup.type}:`);
    console.log(`   原始项目 [${dup.original.index}]: ${dup.original.title}`);
    console.log(`   重复项目 [${dup.duplicate.index}]: ${dup.duplicate.title}`);
    console.log(`   URL: ${dup.duplicate.url}`);
    console.log('');
  });
  
  // 统计重复情况
  const urlDups = duplicates.filter(d => d.type === 'URL重复').length;
  const titleDups = duplicates.filter(d => d.type === '标题重复').length;
  
  console.log(`\n统计信息:`);
  console.log(`- 总订阅源数量: ${feedsData.length}`);
  console.log(`- URL重复: ${urlDups} 个`);
  console.log(`- 标题重复: ${titleDups} 个`);
  console.log(`- 总重复项: ${duplicates.length} 个`);
  
  // 建议清理
  console.log(`\n建议:`);
  console.log(`1. 根据URL去重（最可靠）`);
  console.log(`2. 保留第一个出现的订阅源`);
  console.log(`3. 清理后将订阅源数量减少到 ${feedsData.length - urlDups} 个`);
} else {
  console.log('✅ 未发现重复的订阅源');
}

// 显示一些统计信息
console.log(`\n订阅源统计:`);
console.log(`- 国内订阅源: ${feedsData.filter(f => f.category === 'domestic').length} 个`);
console.log(`- 国外订阅源: ${feedsData.filter(f => f.category === 'international').length} 个`);
console.log(`- 未分类订阅源: ${feedsData.filter(f => !f.category).length} 个`);

// 检查一些可能的重复模式
console.log(`\n检查可能的重复模式:`);
const domains = feedsData.map(f => {
  try {
    return new URL(f.url).hostname;
  } catch {
    return 'invalid-url';
  }
});

const domainCounts = domains.reduce((acc, domain) => {
  acc[domain] = (acc[domain] || 0) + 1;
  return acc;
}, {});

const commonDomains = Object.entries(domainCounts)
  .filter(([_, count]) => count > 1)
  .sort((a, b) => b[1] - a[1]);

if (commonDomains.length > 0) {
  console.log('\n常见域名（出现多次）:');
  commonDomains.forEach(([domain, count]) => {
    console.log(`   ${domain}: ${count} 次`);
  });
}