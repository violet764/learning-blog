# 数据库学习笔记

本模块涵盖主流数据库和消息队列技术的现代用法与最佳实践。

## 关系型数据库

### MySQL

MySQL 是世界上最流行的开源关系型数据库，广泛应用于 Web 应用和企业级系统。

- [MySQL 基础与最佳实践](./mysql-basics.md) - 安装配置、SQL 基础、性能优化
- [MySQL 高级特性](./mysql-advanced.md) - 索引优化、事务、锁机制、复制与高可用

### PostgreSQL

PostgreSQL 是功能最强大的开源关系型数据库，以其扩展性和标准兼容性著称。

- [PostgreSQL 基础与最佳实践](./postgresql-basics.md) - 数据类型、SQL 特性、性能调优
- [PostgreSQL 高级特性](./postgresql-advanced.md) - JSON 支持、全文搜索、扩展生态

## NoSQL 数据库

### Redis

Redis 是高性能的内存数据结构存储，常用于缓存、会话管理和实时数据处理。

- [Redis 基础与实践](./redis-basics.md) - 数据结构、持久化、集群部署
- [Redis 高级模式](./redis-advanced.md) - 分布式锁、发布订阅、最佳实践

## 消息队列

### Apache Kafka

Kafka 是分布式流处理平台，用于构建实时数据管道和流式应用。

- [Kafka 基础与架构](./kafka-basics.md) - 核心概念、部署配置、生产消费模型
- [Kafka 高级实践](./kafka-advanced.md) - 流处理、安全配置、性能调优

### RabbitMQ

RabbitMQ 是可靠的消息代理，支持多种消息协议，适合复杂路由场景。

- [RabbitMQ 基础与实践](./rabbitmq-basics.md) - AMQP 协议、队列模型、交换机类型
- [RabbitMQ 高级模式](./rabbitmq-advanced.md) - 死信队列、延迟消息、高可用配置

## 学习路径

```
关系型数据库基础 → NoSQL 数据库 → 消息队列 → 分布式系统设计
```

## 参考资料

- [MySQL 官方文档](https://dev.mysql.com/doc/)
- [PostgreSQL 官方文档](https://www.postgresql.org/docs/)
- [Redis 官方文档](https://redis.io/docs/)
- [Apache Kafka 官方文档](https://kafka.apache.org/documentation/)
- [RabbitMQ 官方文档](https://www.rabbitmq.com/docs/)
