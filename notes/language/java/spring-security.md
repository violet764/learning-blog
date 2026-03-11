# Spring Security 安全框架

> Spring Security 是 Spring 生态中的安全框架，提供认证（Authentication）和授权（Authorization）功能，是构建安全 Java 应用的首选方案。

## 应用安全

### 安全框架

```
没有安全保护的 Web 应用：

用户 → 发送请求 → 服务器 → 直接返回数据

问题：
1. 任何人都能访问所有接口（包括管理员接口）
2. 没有登录就能看到个人数据
3. 无法区分不同用户的权限
4. 容易被攻击（CSRF、XSS 等）

有安全保护的 Web 应用：

用户 → 发送请求 → 安全过滤器 → 验证身份和权限 → 服务器 → 返回数据
                           ↓
                    身份验证失败 → 拒绝访问
```

### 认证 vs 授权

```
认证（Authentication）：你是谁？
├── 用户名 + 密码登录
├── 手机验证码登录
├── 第三方登录（微信、GitHub）
└── Token 验证（JWT）

授权（Authorization）：你能做什么？
├── 普通用户：查看个人信息
├── 管理员：管理所有用户
└── 超级管理员：系统配置

举例：
认证：门卫检查你的工牌，确认你是张三
授权：张三是普通员工，只能进入办公室，不能进入机房
```

### Spring Security 核心概念

```
┌─────────────────────────────────────────────────────┐
│                  Spring Security 架构                │
├─────────────────────────────────────────────────────┤
│                                                     │
│  HTTP 请求                                           │
│     ↓                                               │
│  ┌─────────────────┐                                │
│  │ Security Filter │  ← 一系列过滤器链               │
│  │     Chain       │                                │
│  └────────┬────────┘                                │
│           ↓                                         │
│  ┌─────────────────┐                                │
│  │ Authentication  │  ← 认证：验证用户身份           │
│  │    Manager      │                                │
│  └────────┬────────┘                                │
│           ↓                                         │
│  ┌─────────────────┐                                │
│  │ User Details    │  ← 用户详情：用户名、密码、权限  │
│  │    Service      │                                │
│  └────────┬────────┘                                │
│           ↓                                         │
│  ┌─────────────────┐                                │
│  │ Security        │  ← 安全上下文：存储认证信息      │
│  │   Context       │                                │
│  └────────┬────────┘                                │
│           ↓                                         │
│  ┌─────────────────┐                                │
│  │ Authorization   │  ← 授权：检查用户权限           │
│  │   Decision      │                                │
│  └────────┬────────┘                                │
│           ↓                                         │
│  访问资源 或 拒绝访问                                │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 框架概述

Spring Security 是一个强大且高度可定制的身份验证和访问控制框架，专注于为 Java 应用程序提供声明式安全服务。

### 核心功能

| 功能 | 说明 | 类比 |
|------|------|------|
| **认证** | 确认用户身份（你是谁） | 门卫检查工牌 |
| **授权** | 确认用户权限（你能做什么） | 进入不同房间需要不同权限 |
| **CSRF 防护** | 防止跨站请求伪造攻击 | 验证请求来源 |
| **Session 管理** | 会话固定保护、并发控制 | 管理登录状态 |
| **密码加密** | BCrypt、PBKDF2 等加密算法 | 保护密码安全 |
| **安全头** | X-Frame-Options、XSS 保护等 | 防止常见攻击 |

### 认证流程详解

```
用户请求
    ↓
SecurityContextPersistenceFilter
    │ 从 Session 中获取已有的安全上下文
    ↓
UsernamePasswordAuthenticationFilter（表单登录认证）
    │ 拦截登录请求，提取用户名密码
    ↓
AuthenticationManager（认证管理器）
    │ 协调认证过程
    ↓
AuthenticationProvider（认证提供者）
    │ 具体的认证逻辑
    ↓
UserDetailsService（加载用户信息）
    │ 从数据库加载用户详情
    ↓
Password Authentication
    │ 验证密码是否正确
    ↓
┌─────────────────────────────┐
│ 认证成功                     │ 认证失败
│     ↓                       │     ↓
│ SecurityContextHolder       │ AuthenticationFailureHandler
│ （存储认证信息）              │ （处理失败逻辑）
│     ↓                       │
│ 访问资源                     │
└─────────────────────────────┘
```

---

## 快速开始

### 添加依赖

```xml
<!-- pom.xml -->
<!-- Spring Boot Security Starter -->
<!-- 添加这个依赖后，Spring Security 自动配置就会生效 -->
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-security</artifactId>
</dependency>

<!-- 如果需要 JWT 支持 -->
<dependency>
    <groupId>io.jsonwebtoken</groupId>
    <artifactId>jjwt-api</artifactId>
    <version>0.12.3</version>
</dependency>
<dependency>
    <groupId>io.jsonwebtoken</groupId>
    <artifactId>jjwt-impl</artifactId>
    <version>0.12.3</version>
    <scope>runtime</scope>
</dependency>
<dependency>
    <groupId>io.jsonwebtoken</groupId>
    <artifactId>jjwt-jackson</artifactId>
    <version>0.12.3</version>
    <scope>runtime</scope>
</dependency>
```

### 默认行为（了解即可）

添加依赖后，Spring Security 默认配置：

```
默认行为：
├── 所有请求都需要认证
├── 提供默认登录页面 /login
├── 生成随机密码（控制台输出）
│   例如：Using generated security password: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
├── 默认用户名 user
└── 默认登出地址 /logout

控制台输出示例：
Using generated security password: 8e55f5b4-3c2a-4d1e-9f8b-7c6d5e4f3a2b
```

### 基础配置示例

```java
package com.example.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.core.userdetails.User;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.security.provisioning.InMemoryUserDetailsManager;
import org.springframework.security.web.SecurityFilterChain;

/**
 * Spring Security 配置类
 * 
 * @Configuration：标记这是一个配置类
 * @EnableWebSecurity：启用 Web 安全功能
 */
@Configuration
@EnableWebSecurity
public class SecurityConfig {
    
    /**
     * 配置安全过滤器链
     * 
     * 这是 Spring Security 的核心配置
     * 定义哪些 URL 需要认证、哪些可以公开访问
     */
    @Bean
    public SecurityFilterChain securityFilterChain(HttpSecurity http) throws Exception {
        http
            // ========== 授权配置 ==========
            // 定义 URL 访问规则
            .authorizeHttpRequests(auth -> auth
                // permitAll()：允许所有人访问（包括未登录用户）
                .requestMatchers("/public/**").permitAll()      // 公开资源
                .requestMatchers("/api/auth/**").permitAll()    // 登录注册接口
                
                // hasRole()：需要指定角色才能访问
                .requestMatchers("/admin/**").hasRole("ADMIN")  // 需要 ADMIN 角色
                
                // hasAnyRole()：需要任意一个角色
                .requestMatchers("/user/**").hasAnyRole("USER", "ADMIN")
                
                // anyRequest()：其他所有请求
                .anyRequest().authenticated()  // 需要登录才能访问
            )
            
            // ========== 表单登录配置 ==========
            .formLogin(form -> form
                // 自定义登录页面 URL
                .loginPage("/login")
                // 允许所有人访问登录页面
                .permitAll()
                // 登录成功后的默认跳转地址
                .defaultSuccessUrl("/home")
                // 登录失败后的跳转地址
                .failureUrl("/login?error")
            )
            
            // ========== 注销配置 ==========
            .logout(logout -> logout
                // 注销 URL
                .logoutUrl("/logout")
                // 注销成功后跳转
                .logoutSuccessUrl("/login?logout")
                // 允许所有人访问注销
                .permitAll()
                // 注销时使 Session 失效
                .invalidateHttpSession(true)
                // 清除认证信息
                .clearAuthentication(true)
            );
        
        return http.build();
    }
    
    /**
     * 密码加密器
     * 
     * BCrypt 是一种安全的密码哈希算法：
     * 1. 自动加盐（salt），相同的密码生成不同的哈希值
     * 2. 计算成本可调，抵抗暴力破解
     * 3. 是目前推荐的密码存储方式
     */
    @Bean
    public PasswordEncoder passwordEncoder() {
        // BCryptPasswordEncoder 默认强度为 10
        // 强度越大，计算越慢，越安全
        return new BCryptPasswordEncoder();
    }
    
    /**
     * 内存用户（仅用于测试，生产环境请使用数据库）
     * 
     * 实际项目中，用户信息存储在数据库
     * 这里仅作为示例
     */
    @Bean
    public UserDetailsService userDetailsService(PasswordEncoder passwordEncoder) {
        // 创建普通用户
        UserDetails user = User.builder()
            .username("user")           // 用户名
            .password(passwordEncoder.encode("123456"))  // 加密后的密码
            .roles("USER")              // 角色（会自动添加 ROLE_ 前缀）
            .build();
        
        // 创建管理员用户
        UserDetails admin = User.builder()
            .username("admin")
            .password(passwordEncoder.encode("123456"))
            .roles("ADMIN", "USER")     // 管理员可以有两个角色
            .build();
        
        // 返回内存用户管理器
        return new InMemoryUserDetailsManager(user, admin);
    }
}
```

---

## 认证机制详解

### 密码加密

```java
package com.example.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;

@Configuration
public class SecurityConfig {
    
    /**
     * 密码加密器 Bean
     * 
     * BCrypt 加密特点：
     * 1. 同一个密码，每次加密结果不同（因为有随机盐）
     * 2. 加密结果长度固定为 60 个字符
     * 3. 可以通过 matches 方法验证密码
     */
    @Bean
    public PasswordEncoder passwordEncoder() {
        // 创建 BCrypt 加密器
        // 参数是 strength（强度），默认 10
        // 强度越高越安全，但计算越慢
        return new BCryptPasswordEncoder(10);
    }
}

// ==================== 使用示例 ====================

@Service
@RequiredArgsConstructor
public class UserService {
    
    private final UserRepository userRepository;
    private final PasswordEncoder passwordEncoder;
    
    /**
     * 用户注册
     */
    public void register(UserRegisterDTO dto) {
        // 检查用户名是否已存在
        if (userRepository.existsByUsername(dto.getUsername())) {
            throw new RuntimeException("用户名已存在");
        }
        
        User user = new User();
        user.setUsername(dto.getUsername());
        
        // ========== 加密密码 ==========
        // 永远不要存储明文密码！
        // passwordEncoder.encode() 会：
        // 1. 生成随机盐
        // 2. 将盐和密码组合
        // 3. 进行哈希计算
        String encodedPassword = passwordEncoder.encode(dto.getPassword());
        user.setPassword(encodedPassword);
        
        // 示例：
        // 输入密码：123456
        // 加密后：$2a$10$N.zmdr9k7uOCQb376NoUnuTJ8iAt6Z5EHsM8lE9lBOsl7iAt.8F.u
        
        userRepository.save(user);
    }
    
    /**
     * 验证密码
     */
    public boolean checkPassword(Long userId, String rawPassword) {
        User user = userRepository.findById(userId)
            .orElseThrow(() -> new RuntimeException("用户不存在"));
        
        // ========== 验证密码 ==========
        // matches 方法会：
        // 1. 从加密密码中提取盐
        // 2. 用相同的盐加密输入的密码
        // 3. 比较两个哈希值
        return passwordEncoder.matches(rawPassword, user.getPassword());
    }
}
```

### 自定义 UserDetailsService

```java
package com.example.security;

import com.example.entity.User;
import com.example.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.stream.Collectors;

/**
 * 自定义用户详情服务
 * 
 * Spring Security 在认证时会调用这个类
 * 从数据库加载用户信息
 */
@Service
@RequiredArgsConstructor
public class CustomUserDetailsService implements UserDetailsService {
    
    private final UserRepository userRepository;
    
    /**
     * 根据用户名加载用户详情
     * 
     * 这个方法在用户登录时被 Spring Security 自动调用
     * 返回的 UserDetails 包含：
     * - 用户名
     * - 密码（加密后）
     * - 权限列表
     * - 账号状态
     */
    @Override
    public UserDetails loadUserByUsername(String username) throws UsernameNotFoundException {
        // 从数据库查询用户
        User user = userRepository.findByUsername(username)
            .orElseThrow(() -> new UsernameNotFoundException("用户不存在: " + username));
        
        // 构建权限列表
        // 权限格式：ROLE_角色名 或 权限标识
        // 例如：ROLE_ADMIN, user:delete, user:edit
        List<GrantedAuthority> authorities = user.getRoles().stream()
            .map(role -> new SimpleGrantedAuthority("ROLE_" + role.getName()))
            .collect(Collectors.toList());
        
        // 添加具体权限
        user.getPermissions().forEach(permission -> 
            authorities.add(new SimpleGrantedAuthority(permission.getName()))
        );
        
        // 返回 Spring Security 的 UserDetails 实现
        return new org.springframework.security.core.userdetails.User(
            user.getUsername(),           // 用户名
            user.getPassword(),           // 密码（加密后）
            user.isEnabled(),             // 账号是否启用
            true,                         // 账号是否未过期
            true,                         // 凭证（密码）是否未过期
            true,                         // 账号是否未锁定
            authorities                   // 权限列表
        );
    }
}

// ==================== 用户实体示例 ====================

@Entity
@Table(name = "sys_user")
@Data
public class User {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    
    private String username;
    private String password;
    private Boolean enabled = true;
    private Boolean accountNonExpired = true;
    private Boolean accountNonLocked = true;
    private Boolean credentialsNonExpired = true;
    
    // 用户角色（多对多关系）
    @ManyToMany(fetch = FetchType.EAGER)
    @JoinTable(
        name = "sys_user_role",
        joinColumns = @JoinColumn(name = "user_id"),
        inverseJoinColumns = @JoinColumn(name = "role_id")
    )
    private Set<Role> roles;
    
    // 用户权限
    @Transient
    private Set<Permission> permissions;
}
```

### 自定义登录成功/失败处理

```java
package com.example.security.handler;

import com.example.common.Result;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.RequiredArgsConstructor;
import org.springframework.security.core.Authentication;
import org.springframework.security.web.authentication.AuthenticationSuccessHandler;
import org.springframework.stereotype.Component;

import java.io.IOException;

/**
 * 登录成功处理器
 * 
 * 当用户认证成功后，这个类会被调用
 * 可以在这里：
 * - 返回 JSON 响应（前后端分离）
 * - 生成 Token
 * - 记录登录日志
 */
@Component
@RequiredArgsConstructor
public class LoginSuccessHandler implements AuthenticationSuccessHandler {
    
    private final ObjectMapper objectMapper;  // JSON 序列化工具
    
    @Override
    public void onAuthenticationSuccess(
            HttpServletRequest request, 
            HttpServletResponse response,
            Authentication authentication) throws IOException {
        
        // authentication 包含认证成功后的用户信息
        // authentication.getName() 获取用户名
        // authentication.getAuthorities() 获取权限列表
        
        // 设置响应类型
        response.setContentType("application/json;charset=UTF-8");
        
        // 返回成功响应
        Result<Void> result = Result.success("登录成功");
        response.getWriter().write(objectMapper.writeValueAsString(result));
        
        // 如果是 JWT 认证，这里可以生成 Token 并返回
        // String token = jwtUtils.generateToken(authentication);
        // Result<LoginVO> result = Result.success(new LoginVO(token));
    }
}

// ==================== 登录失败处理器 ====================

@Component
@RequiredArgsConstructor
public class LoginFailureHandler implements AuthenticationFailureHandler {
    
    private final ObjectMapper objectMapper;
    
    @Override
    public void onAuthenticationFailure(
            HttpServletRequest request, 
            HttpServletResponse response,
            AuthenticationException exception) throws IOException {
        
        response.setContentType("application/json;charset=UTF-8");
        response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
        
        // 根据不同的异常类型返回不同的错误信息
        String message;
        if (exception instanceof BadCredentialsException) {
            message = "用户名或密码错误";
        } else if (exception instanceof DisabledException) {
            message = "账号已被禁用";
        } else if (exception instanceof LockedException) {
            message = "账号已被锁定";
        } else if (exception instanceof AccountExpiredException) {
            message = "账号已过期";
        } else if (exception instanceof CredentialsExpiredException) {
            message = "密码已过期";
        } else {
            message = "登录失败: " + exception.getMessage();
        }
        
        Result<Void> result = Result.error(401, message);
        response.getWriter().write(objectMapper.writeValueAsString(result));
    }
}

// ==================== 配置中使用 ====================

@Configuration
@EnableWebSecurity
@RequiredArgsConstructor
public class SecurityConfig {
    
    private final LoginSuccessHandler loginSuccessHandler;
    private final LoginFailureHandler loginFailureHandler;
    
    @Bean
    public SecurityFilterChain securityFilterChain(HttpSecurity http) throws Exception {
        http.formLogin(form -> form
            // 登录接口地址
            .loginProcessingUrl("/api/auth/login")
            // 成功处理器
            .successHandler(loginSuccessHandler)
            // 失败处理器
            .failureHandler(loginFailureHandler)
        );
        return http.build();
    }
}
```

---

## JWT 认证（前后端分离必备）

### 什么是 JWT？

```
JWT（JSON Web Token）是一种开放标准，用于在各方之间安全传输信息。

JWT 结构：xxxxx.yyyyy.zzzzz
         ↑       ↑      ↑
       Header  Payload  Signature

Header（头部）：令牌类型和签名算法
{
  "alg": "HS256",
  "typ": "JWT"
}

Payload（载荷）：用户信息（不敏感数据）
{
  "sub": "zhangsan",     // 主题（用户名）
  "iat": 1516239022,     // 签发时间
  "exp": 1516242622,     // 过期时间
  "roles": ["USER"]      // 自定义数据
}

Signature（签名）：防止数据被篡改
HMACSHA256(
  base64UrlEncode(header) + "." + base64UrlEncode(payload),
  secret
)

JWT 工作流程：
┌──────────┐         ┌──────────┐         ┌──────────┐
│  客户端   │         │  服务器   │         │  数据库   │
└────┬─────┘         └────┬─────┘         └────┬─────┘
     │  1. 登录请求       │                    │
     │ ─────────────────→│                    │
     │                   │  2. 验证用户        │
     │                   │ ──────────────────→│
     │                   │  3. 返回用户信息     │
     │                   │ ←──────────────────│
     │                   │                    │
     │  4. 生成 JWT 返回  │                    │
     │ ←─────────────────│                    │
     │                   │                    │
     │  5. 携带 JWT 访问  │                    │
     │ ─────────────────→│                    │
     │                   │  6. 验证 JWT        │
     │                   │  （无需查询数据库）  │
     │  7. 返回数据       │                    │
     │ ←─────────────────│                    │
```

### JWT 工具类

```java
package com.example.security;

import io.jsonwebtoken.*;
import io.jsonwebtoken.security.Keys;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import javax.crypto.SecretKey;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * JWT 工具类
 * 
 * 负责生成、解析、验证 JWT Token
 */
@Component
public class JwtUtils {
    
    // 密钥（从配置文件读取，至少 256 位）
    @Value("${jwt.secret:your-secret-key-must-be-at-least-256-bits-long}")
    private String secret;
    
    // 过期时间（毫秒），默认 24 小时
    @Value("${jwt.expiration:86400000}")
    private long expiration;
    
    // Token 前缀
    public static final String TOKEN_PREFIX = "Bearer ";
    
    // 请求头名称
    public static final String HEADER_NAME = "Authorization";
    
    /**
     * 获取签名密钥
     */
    private SecretKey getSignKey() {
        // 使用 HMAC-SHA 算法的密钥
        return Keys.hmacShaKeyFor(secret.getBytes());
    }
    
    /**
     * 生成 JWT Token
     * 
     * @param username 用户名
     * @param roles 角色列表
     * @return JWT Token
     */
    public String generateToken(String username, List<String> roles) {
        // 自定义声明（可以放任何非敏感数据）
        Map<String, Object> claims = new HashMap<>();
        claims.put("roles", roles);
        
        return Jwts.builder()
            // 设置声明
            .claims(claims)
            // 设置主题（通常是用户名）
            .subject(username)
            // 设置签发时间
            .issuedAt(new Date())
            // 设置过期时间
            .expiration(new Date(System.currentTimeMillis() + expiration))
            // 使用密钥签名
            .signWith(getSignKey())
            // 生成 Token
            .compact();
    }
    
    /**
     * 解析 JWT Token
     * 
     * @param token JWT Token
     * @return Claims（声明集合）
     */
    public Claims parseToken(String token) {
        return Jwts.parser()
            // 设置验证密钥
            .verifyWith(getSignKey())
            .build()
            // 解析 Token
            .parseSignedClaims(token)
            // 获取声明
            .getPayload();
    }
    
    /**
     * 验证 Token 是否有效
     */
    public boolean validateToken(String token) {
        try {
            parseToken(token);
            return true;
        } catch (ExpiredJwtException e) {
            // Token 已过期
            System.out.println("Token 已过期");
        } catch (UnsupportedJwtException e) {
            // 不支持的 Token
            System.out.println("不支持的 Token");
        } catch (MalformedJwtException e) {
            // Token 格式错误
            System.out.println("Token 格式错误");
        } catch (SignatureException e) {
            // 签名验证失败
            System.out.println("签名验证失败");
        } catch (IllegalArgumentException e) {
            // Token 为空
            System.out.println("Token 为空");
        }
        return false;
    }
    
    /**
     * 从 Token 获取用户名
     */
    public String getUsername(String token) {
        return parseToken(token).getSubject();
    }
    
    /**
     * 从 Token 获取角色列表
     */
    @SuppressWarnings("unchecked")
    public List<String> getRoles(String token) {
        return parseToken(token).get("roles", List.class);
    }
    
    /**
     * 检查 Token 是否即将过期（可用于刷新 Token）
     */
    public boolean isTokenExpiringSoon(String token) {
        Date expiration = parseToken(token).getExpiration();
        // 如果距离过期时间少于 30 分钟，认为即将过期
        return expiration.getTime() - System.currentTimeMillis() < 30 * 60 * 1000;
    }
}
```

### JWT 认证过滤器

```java
package com.example.security;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.RequiredArgsConstructor;
import org.springframework.lang.NonNull;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.web.authentication.WebAuthenticationDetailsSource;
import org.springframework.stereotype.Component;
import org.springframework.util.StringUtils;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;

/**
 * JWT 认证过滤器
 * 
 * 拦截每个请求，从请求头提取 JWT Token
 * 验证 Token 并设置认证信息
 * 
 * OncePerRequestFilter：确保每个请求只过滤一次
 */
@Component
@RequiredArgsConstructor
public class JwtAuthenticationFilter extends OncePerRequestFilter {
    
    private final JwtUtils jwtUtils;
    private final UserDetailsService userDetailsService;
    
    @Override
    protected void doFilterInternal(
            @NonNull HttpServletRequest request, 
            @NonNull HttpServletResponse response, 
            @NonNull FilterChain filterChain) throws ServletException, IOException {
        
        try {
            // ========== 1. 从请求头获取 Token ==========
            String token = getTokenFromRequest(request);
            
            // ========== 2. 验证 Token ==========
            if (StringUtils.hasText(token) && jwtUtils.validateToken(token)) {
                
                // 从 Token 获取用户名
                String username = jwtUtils.getUsername(token);
                
                // 加载用户详情
                UserDetails userDetails = userDetailsService.loadUserByUsername(username);
                
                // ========== 3. 创建认证对象 ==========
                // UsernamePasswordAuthenticationToken 是 Authentication 的实现
                // 参数：principal（用户详情）、credentials（凭证，JWT 中不需要）、authorities（权限）
                UsernamePasswordAuthenticationToken authentication = 
                    new UsernamePasswordAuthenticationToken(
                        userDetails, 
                        null, 
                        userDetails.getAuthorities()
                    );
                
                // 设置认证详情（包含请求信息）
                authentication.setDetails(
                    new WebAuthenticationDetailsSource().buildDetails(request)
                );
                
                // ========== 4. 存入安全上下文 ==========
                // SecurityContextHolder 存储当前认证用户的信息
                // 后续可以通过 SecurityContextHolder.getContext().getAuthentication() 获取
                SecurityContextHolder.getContext().setAuthentication(authentication);
            }
        } catch (Exception e) {
            // 认证失败，清除上下文
            SecurityContextHolder.clearContext();
        }
        
        // 继续过滤器链
        filterChain.doFilter(request, response);
    }
    
    /**
     * 从请求头提取 Token
     * 
     * Authorization: Bearer xxxxx.yyyyy.zzzzz
     */
    private String getTokenFromRequest(HttpServletRequest request) {
        String bearerToken = request.getHeader(JwtUtils.HEADER_NAME);
        
        // 检查是否以 "Bearer " 开头
        if (StringUtils.hasText(bearerToken) && bearerToken.startsWith(JwtUtils.TOKEN_PREFIX)) {
            // 去掉 "Bearer " 前缀，返回纯 Token
            return bearerToken.substring(JwtUtils.TOKEN_PREFIX.length());
        }
        
        return null;
    }
}
```

### JWT 配置类

```java
package com.example.config;

import com.example.security.JwtAuthenticationFilter;
import lombok.RequiredArgsConstructor;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.method.configuration.EnableMethodSecurity;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.annotation.web.configurers.AbstractHttpConfigurer;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter;

@Configuration
@EnableWebSecurity
@EnableMethodSecurity  // 启用方法级安全注解
@RequiredArgsConstructor
public class JwtSecurityConfig {
    
    private final JwtAuthenticationFilter jwtAuthenticationFilter;
    
    @Bean
    public SecurityFilterChain securityFilterChain(HttpSecurity http) throws Exception {
        http
            // ========== 禁用 CSRF ==========
            // JWT 不需要 CSRF 保护（因为没有 Cookie）
            .csrf(AbstractHttpConfigurer::disable)
            
            // ========== 配置 Session ==========
            // JWT 是无状态的，不需要 Session
            .sessionManagement(session -> session
                .sessionCreationPolicy(SessionCreationPolicy.STATELESS)
            )
            
            // ========== 授权配置 ==========
            .authorizeHttpRequests(auth -> auth
                // 公开接口
                .requestMatchers("/api/auth/login", "/api/auth/register").permitAll()
                .requestMatchers("/api/public/**").permitAll()
                // 管理员接口
                .requestMatchers("/api/admin/**").hasRole("ADMIN")
                // 其他接口需要认证
                .anyRequest().authenticated()
            )
            
            // ========== 添加 JWT 过滤器 ==========
            // 在 UsernamePasswordAuthenticationFilter 之前执行
            .addFilterBefore(jwtAuthenticationFilter, 
                UsernamePasswordAuthenticationFilter.class);
        
        return http.build();
    }
}
```

### 认证控制器

```java
package com.example.controller;

import com.example.common.Result;
import com.example.dto.LoginDTO;
import com.example.dto.RegisterDTO;
import com.example.security.JwtUtils;
import com.example.vo.LoginVO;
import com.example.vo.UserInfoVO;
import lombok.RequiredArgsConstructor;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.stream.Collectors;

/**
 * 认证控制器
 */
@RestController
@RequestMapping("/api/auth")
@RequiredArgsConstructor
public class AuthController {
    
    private final AuthenticationManager authenticationManager;
    private final JwtUtils jwtUtils;
    
    /**
     * 用户登录
     */
    @PostMapping("/login")
    public Result<LoginVO> login(@RequestBody LoginDTO dto) {
        // ========== 1. 认证 ==========
        // 创建认证 Token（包含用户名和密码）
        UsernamePasswordAuthenticationToken authToken = 
            new UsernamePasswordAuthenticationToken(dto.getUsername(), dto.getPassword());
        
        // 执行认证（会调用 UserDetailsService 和 PasswordEncoder）
        Authentication authentication = authenticationManager.authenticate(authToken);
        
        // ========== 2. 生成 JWT ==========
        // 从认证结果获取用户信息
        UserDetails userDetails = (UserDetails) authentication.getPrincipal();
        
        // 提取角色列表
        List<String> roles = userDetails.getAuthorities().stream()
            .map(GrantedAuthority::getAuthority)
            .collect(Collectors.toList());
        
        // 生成 Token
        String token = jwtUtils.generateToken(userDetails.getUsername(), roles);
        
        // ========== 3. 返回结果 ==========
        return Result.success(new LoginVO(token, userDetails.getUsername(), roles));
    }
    
    /**
     * 获取当前用户信息
     */
    @GetMapping("/info")
    public Result<UserInfoVO> getUserInfo() {
        // 从安全上下文获取当前认证用户
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        
        String username = authentication.getName();
        
        // 返回用户信息（实际项目中从数据库获取完整信息）
        return Result.success(new UserInfoVO(username));
    }
    
    /**
     * 用户登出
     * 
     * JWT 是无状态的，服务端不需要处理登出
     * 客户端只需删除 Token 即可
     */
    @PostMapping("/logout")
    public Result<Void> logout() {
        return Result.success();
    }
}
```

---

## 授权机制详解

### 基于角色的授权

```java
@Configuration
@EnableWebSecurity
public class SecurityConfig {
    
    @Bean
    public SecurityFilterChain securityFilterChain(HttpSecurity http) throws Exception {
        http.authorizeHttpRequests(auth -> auth
            // ========== 角色检查 ==========
            // hasRole：需要单个角色（会自动添加 ROLE_ 前缀）
            // .hasRole("ADMIN") 实际检查的是 ROLE_ADMIN
            .requestMatchers("/admin/**").hasRole("ADMIN")
            
            // hasAnyRole：需要任意一个角色
            .requestMatchers("/user/**").hasAnyRole("USER", "ADMIN")
            
            // ========== 权限检查 ==========
            // hasAuthority：检查具体权限（不会添加前缀）
            // 通常用于细粒度权限控制
            .requestMatchers("/user/delete/**").hasAuthority("user:delete")
            .requestMatchers("/user/edit/**").hasAnyAuthority("user:edit", "user:admin")
            
            // ========== IP 地址限制 ==========
            .requestMatchers("/internal/**")
                .hasIpAddress("192.168.1.0/24")
        );
        return http.build();
    }
}
```

### 方法级安全（注解方式）

```java
@Configuration
@EnableMethodSecurity  // 启用方法级安全
public class SecurityConfig { }

// ==================== 使用示例 ====================

@Service
public class UserService {
    
    /**
     * @PreAuthorize：方法执行前检查权限
     * 
     * 表达式说明：
     * hasRole('ADMIN')：是否有 ADMIN 角色
     * hasAuthority('user:delete')：是否有 user:delete 权限
     * isAuthenticated()：是否已登录
     * #id：方法参数（SpEL 表达式）
     * authentication.principal：当前登录用户
     */
    @PreAuthorize("hasRole('ADMIN')")
    public void deleteUser(Long id) {
        // 只有管理员能删除用户
    }
    
    /**
     * 复杂权限表达式
     * 
     * 管理员可以删除任何用户
     * 普通用户只能删除自己
     */
    @PreAuthorize("hasRole('ADMIN') or #id == authentication.principal.id")
    public void deleteAccount(Long id) {
        // 删除账号
    }
    
    /**
     * @PostAuthorize：方法执行后检查权限
     * 
     * returnObject：方法的返回值
     * 只有返回的用户是当前用户本人时才允许访问
     */
    @PostAuthorize("returnObject.username == authentication.name")
    public User getUserProfile(Long id) {
        return userRepository.findById(id);
    }
    
    /**
     * @PreFilter：过滤方法参数
     * 
     * filterObject：集合中的每个元素
     * 过滤掉不是当前用户创建的数据
     */
    @PreFilter("filterObject.createdBy == authentication.name")
    public void batchDelete(List<Article> articles) {
        // 只删除当前用户创建的文章
    }
    
    /**
     * @PostFilter：过滤返回结果
     * 
     * 自动过滤返回集合中的元素
     */
    @PostFilter("filterObject.status == 'PUBLISHED'")
    public List<Article> getAllArticles() {
        return articleRepository.findAll();
    }
}
```

### 权限注解对比

| 注解 | 执行时机 | 用途 |
|------|----------|------|
| `@PreAuthorize` | 方法执行前 | 检查是否有权限执行 |
| `@PostAuthorize` | 方法执行后 | 根据返回值决定是否允许 |
| `@PreFilter` | 方法执行前 | 过滤输入参数 |
| `@PostFilter` | 方法执行后 | 过滤返回结果 |

### 常用 SpEL 表达式

```java
// ========== 角色和权限 ==========
hasRole('ADMIN')              // 有 ADMIN 角色
hasAnyRole('ADMIN', 'USER')   // 有任意一个角色
hasAuthority('user:delete')   // 有 user:delete 权限
hasAnyAuthority('user:read', 'user:write')

// ========== 认证状态 ==========
isAuthenticated()   // 已登录
isAnonymous()       // 未登录
isRememberMe()      // 记住我登录
isFullyAuthenticated()  // 完整认证（非记住我）

// ========== 访问当前用户 ==========
authentication          // 认证对象
authentication.name     // 用户名
authentication.principal  // UserDetails 对象
authentication.authorities  // 权限集合

// ========== 逻辑运算 ==========
and, or, not        // 与、或、非
#param              // 方法参数
returnObject        // 返回值（@PostAuthorize）
filterObject        // 集合元素（@PreFilter/@PostFilter）
```

---

## 常见配置

### CORS 跨域配置

```java
@Configuration
@EnableWebSecurity
public class SecurityConfig {
    
    @Bean
    public SecurityFilterChain securityFilterChain(HttpSecurity http) throws Exception {
        http.cors(cors -> cors.configurationSource(corsConfigurationSource()));
        // ... 其他配置
        return http.build();
    }
    
    /**
     * CORS 配置
     */
    @Bean
    public CorsConfigurationSource corsConfigurationSource() {
        CorsConfiguration configuration = new CorsConfiguration();
        
        // 允许的源（前端地址）
        configuration.setAllowedOrigins(Arrays.asList(
            "http://localhost:3000",
            "http://localhost:8080"
        ));
        
        // 允许的 HTTP 方法
        configuration.setAllowedMethods(Arrays.asList(
            "GET", "POST", "PUT", "DELETE", "OPTIONS"
        ));
        
        // 允许的请求头
        configuration.setAllowedHeaders(Arrays.asList("*"));
        
        // 允许携带凭证（Cookie、Authorization）
        configuration.setAllowCredentials(true);
        
        // 预检请求缓存时间（秒）
        configuration.setMaxAge(3600L);
        
        // 注册配置
        UrlBasedCorsConfigurationSource source = new UrlBasedCorsConfigurationSource();
        source.registerCorsConfiguration("/**", configuration);
        
        return source;
    }
}
```

### CSRF 配置

```java
// CSRF（跨站请求伪造）攻击原理：
// 1. 用户登录了网站 A
// 2. 用户访问了恶意网站 B
// 3. 网站 B 向网站 A 发送请求（携带用户 Cookie）
// 4. 网站 A 认为是用户操作

// ========== 禁用 CSRF（JWT 推荐）==========
http.csrf(csrf -> csrf.disable());

// ========== 启用 CSRF（传统 Session 方式）==========
http.csrf(csrf -> csrf
    // 使用 Cookie 存储 CSRF Token
    .csrfTokenRepository(CookieCsrfTokenRepository.withHttpOnlyFalse())
    // 忽略某些接口
    .ignoringRequestMatchers("/api/public/**")
);
```

### Session 配置

```java
http.sessionManagement(session -> session
    // ========== Session 创建策略 ==========
    // STATELESS：不创建 Session（JWT 推荐）
    // IF_REQUIRED：需要时创建（默认）
    // ALWAYS：总是创建
    // NEVER：不创建，但使用已有的
    .sessionCreationPolicy(SessionCreationPolicy.STATELESS)
    
    // ========== Session 并发控制 ==========
    .maximumSessions(1)  // 每个用户只允许一个 Session
    .maxSessionsPreventsLogin(true)  // 新登录阻止旧登录
    // .maxSessionsPreventsLogin(false)  // 新登录踢出旧登录（默认）
    .expiredUrl("/login?expired")  // 被踢出后跳转
    
    // ========== Session 固定攻击防护 ==========
    .sessionFixation().migrateSession()  // 登录后创建新 Session
);
```

---

## 安全最佳实践

### 密码安全

```java
// ✅ 正确做法
// 1. 使用强加密算法
@Bean
public PasswordEncoder passwordEncoder() {
    return new BCryptPasswordEncoder();  // 或 Argon2、SCrypt
}

// 2. 永远不要存储明文密码
user.setPassword(passwordEncoder.encode(rawPassword));

// 3. 验证密码
boolean valid = passwordEncoder.matches(rawPassword, encodedPassword);

// ❌ 错误做法
// 不要使用 MD5、SHA1 等弱哈希算法
user.setPassword(MD5(password));  // 不安全！
```

### 敏感信息保护

```yaml
# application.yml

# ❌ 不要在配置文件中硬编码敏感信息
spring:
  datasource:
    password: mypassword123  # 危险！

# ✅ 使用环境变量
spring:
  datasource:
    password: ${DB_PASSWORD}

# JWT 密钥
jwt:
  secret: ${JWT_SECRET:your-default-secret-for-development-only}
  expiration: 86400000  # 24小时
```

### HTTPS 强制

```java
// 生产环境强制 HTTPS
http.requiresChannel(channel -> channel
    .anyRequest().requiresSecure()
);
```

### 安全头配置

```java
http.headers(headers -> headers
    // 防止点击劫持
    .frameOptions(HeadersConfigurer.FrameOptionsConfig::deny)
    // XSS 保护
    .xssProtection(xss -> xss.headerValue(XXssProtectionHeaderWriter.HeaderValue.ENABLED_MODE_BLOCK))
    // 禁止 MIME 类型嗅探
    .contentTypeOptions(Customizer.withDefaults())
    // HSTS（强制 HTTPS）
    .httpStrictTransportSecurity(hsts -> hsts
        .includeSubDomains(true)
        .maxAgeInSeconds(31536000)
    )
);
```

---

## 完整配置示例

```java
package com.example.config;

import com.example.security.JwtAuthenticationFilter;
import com.example.security.handler.LoginFailureHandler;
import com.example.security.handler.LoginSuccessHandler;
import lombok.RequiredArgsConstructor;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.dao.DaoAuthenticationProvider;
import org.springframework.security.config.annotation.method.configuration.EnableMethodSecurity;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.annotation.web.configurers.AbstractHttpConfigurer;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter;

/**
 * Spring Security 完整配置
 */
@Configuration
@EnableWebSecurity
@EnableMethodSecurity
@RequiredArgsConstructor
public class SecurityConfig {
    
    private final JwtAuthenticationFilter jwtAuthenticationFilter;
    private final LoginSuccessHandler loginSuccessHandler;
    private final LoginFailureHandler loginFailureHandler;
    private final UserDetailsService userDetailsService;
    
    @Bean
    public SecurityFilterChain securityFilterChain(HttpSecurity http) throws Exception {
        http
            // ========== CSRF ==========
            .csrf(AbstractHttpConfigurer::disable)
            
            // ========== CORS ==========
            .cors(cors -> {})
            
            // ========== Session ==========
            .sessionManagement(session -> 
                session.sessionCreationPolicy(SessionCreationPolicy.STATELESS))
            
            // ========== 授权规则 ==========
            .authorizeHttpRequests(auth -> auth
                .requestMatchers("/api/auth/login", "/api/auth/register").permitAll()
                .requestMatchers("/api/public/**").permitAll()
                .requestMatchers("/api/admin/**").hasRole("ADMIN")
                .anyRequest().authenticated()
            )
            
            // ========== 表单登录 ==========
            .formLogin(form -> form
                .loginProcessingUrl("/api/auth/login")
                .successHandler(loginSuccessHandler)
                .failureHandler(loginFailureHandler)
            )
            
            // ========== JWT 过滤器 ==========
            .addFilterBefore(jwtAuthenticationFilter, 
                UsernamePasswordAuthenticationFilter.class)
            
            // ========== 异常处理 ==========
            .exceptionHandling(exception -> exception
                .authenticationEntryPoint((request, response, authException) -> {
                    response.setContentType("application/json;charset=UTF-8");
                    response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
                    response.getWriter().write("{\"code\":401,\"message\":\"未授权\"}");
                })
                .accessDeniedHandler((request, response, accessDeniedException) -> {
                    response.setContentType("application/json;charset=UTF-8");
                    response.setStatus(HttpServletResponse.SC_FORBIDDEN);
                    response.getWriter().write("{\"code\":403,\"message\":\"权限不足\"}");
                })
            );
        
        return http.build();
    }
    
    @Bean
    public AuthenticationManager authenticationManager() {
        DaoAuthenticationProvider provider = new DaoAuthenticationProvider();
        provider.setUserDetailsService(userDetailsService);
        provider.setPasswordEncoder(passwordEncoder());
        return new org.springframework.security.authentication.ProviderManager(provider);
    }
    
    @Bean
    public PasswordEncoder passwordEncoder() {
        return new BCryptPasswordEncoder();
    }
}
```

---

## 小结

| 概念 | 说明 |
|------|------|
| **认证** | 确认用户身份（登录） |
| **授权** | 确认用户权限（访问控制） |
| **UserDetailsService** | 加载用户详情 |
| **PasswordEncoder** | 密码加密 |
| **SecurityFilterChain** | 安全过滤器链配置 |
| **JWT** | 无状态认证方案 |

### Spring Security 核心要点

1. **认证流程**：Filter → AuthenticationManager → UserDetailsService → SecurityContext
2. **密码加密**：必须使用 BCrypt 等强加密算法
3. **JWT 认证**：适合前后端分离，无状态
4. **方法级安全**：@PreAuthorize 等注解控制权限
5. **安全配置**：CSRF、CORS、Session 等

### 参考资料

- [Spring Security 官方文档](https://docs.spring.io/spring-security/reference/)
- [Spring Security 架构](https://spring.io/guides/topicals/spring-security-architecture)
- [JWT 官网](https://jwt.io/)