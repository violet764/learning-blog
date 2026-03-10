# Spring Security 安全框架

> Spring Security 是 Spring 生态中的安全框架，提供认证（Authentication）和授权（Authorization）功能，是构建安全 Java 应用的首选方案。

## 框架概述

Spring Security 是一个强大且高度可定制的身份验证和访问控制框架，专注于为 Java 应用程序提供声明式安全服务。

### 🎯 核心功能

| 功能 | 说明 |
|------|------|
| **认证** | 确认用户身份（你是谁） |
| **授权** | 确认用户权限（你能做什么） |
| **CSRF 防护** | 防止跨站请求伪造攻击 |
| **Session 管理** | 会话固定保护、并发控制 |
| **密码加密** | BCrypt、PBKDF2 等加密算法 |
| **安全头** | X-Frame-Options、XSS 保护等 |

### 认证流程

```
用户请求
    ↓
SecurityContextPersistenceFilter（获取安全上下文）
    ↓
UsernamePasswordAuthenticationFilter（表单登录认证）
    ↓
AuthenticationManager（认证管理器）
    ↓
AuthenticationProvider（认证提供者）
    ↓
UserDetailsService（加载用户信息）
    ↓
认证成功/失败处理
    ↓
SecurityContextHolder（存储认证信息）
```

---

## 快速开始

### 添加依赖

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-security</artifactId>
</dependency>
```

### 默认行为

添加依赖后，Spring Security 默认：
- 所有请求都需要认证
- 提供默认登录页面 `/login`
- 生成随机密码（控制台输出）
- 默认用户名 `user`

### 基础配置

```java
@Configuration
@EnableWebSecurity
public class SecurityConfig {
    
    @Bean
    public SecurityFilterChain securityFilterChain(HttpSecurity http) throws Exception {
        http
            // 授权配置
            .authorizeHttpRequests(auth -> auth
                .requestMatchers("/public/**").permitAll()      // 公开访问
                .requestMatchers("/admin/**").hasRole("ADMIN")  // 需要 ADMIN 角色
                .anyRequest().authenticated()                   // 其他需要认证
            )
            // 表单登录
            .formLogin(form -> form
                .loginPage("/login")
                .permitAll()
            )
            // 注销
            .logout(logout -> logout
                .logoutUrl("/logout")
                .logoutSuccessUrl("/login?logout")
                .permitAll()
            );
        
        return http.build();
    }
    
    // 内存用户（仅测试使用）
    @Bean
    public UserDetailsService userDetailsService() {
        UserDetails user = User.withDefaultPasswordEncoder()
            .username("user")
            .password("123456")
            .roles("USER")
            .build();
        
        UserDetails admin = User.withDefaultPasswordEncoder()
            .username("admin")
            .password("123456")
            .roles("ADMIN")
            .build();
        
        return new InMemoryUserDetailsManager(user, admin);
    }
}
```

---

## 认证机制

### 密码加密

```java
@Configuration
public class SecurityConfig {
    
    @Bean
    public PasswordEncoder passwordEncoder() {
        return new BCryptPasswordEncoder();  // BCrypt 加密
    }
}

// 使用示例
@Service
public class UserService {
    
    @Autowired
    private PasswordEncoder passwordEncoder;
    
    public void register(UserDTO dto) {
        User user = new User();
        user.setUsername(dto.getUsername());
        // 加密密码
        user.setPassword(passwordEncoder.encode(dto.getPassword()));
        userRepository.save(user);
    }
    
    public boolean checkPassword(String rawPassword, String encodedPassword) {
        return passwordEncoder.matches(rawPassword, encodedPassword);
    }
}
```

### 自定义 UserDetailsService

```java
@Service
public class CustomUserDetailsService implements UserDetailsService {
    
    @Autowired
    private UserRepository userRepository;
    
    @Override
    public UserDetails loadUserByUsername(String username) throws UsernameNotFoundException {
        User user = userRepository.findByUsername(username)
            .orElseThrow(() -> new UsernameNotFoundException("用户不存在: " + username));
        
        // 构建权限列表
        List<GrantedAuthority> authorities = user.getRoles().stream()
            .map(role -> new SimpleGrantedAuthority("ROLE_" + role.getName()))
            .collect(Collectors.toList());
        
        return new org.springframework.security.core.userdetails.User(
            user.getUsername(),
            user.getPassword(),
            user.isEnabled(),     // 账号是否启用
            true,                 // 账号是否未过期
            true,                 // 凭证是否未过期
            true,                 // 账号是否未锁定
            authorities
        );
    }
}
```

### 自定义登录成功/失败处理

```java
@Component
public class LoginSuccessHandler implements AuthenticationSuccessHandler {
    
    @Override
    public void onAuthenticationSuccess(
            HttpServletRequest request, 
            HttpServletResponse response,
            Authentication authentication) throws IOException {
        
        response.setContentType("application/json;charset=UTF-8");
        response.getWriter().write(
            "{\"code\":200,\"message\":\"登录成功\"}"
        );
    }
}

@Component
public class LoginFailureHandler implements AuthenticationFailureHandler {
    
    @Override
    public void onAuthenticationFailure(
            HttpServletRequest request, 
            HttpServletResponse response,
            AuthenticationException exception) throws IOException {
        
        response.setContentType("application/json;charset=UTF-8");
        response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
        response.getWriter().write(
            "{\"code\":401,\"message\":\"" + exception.getMessage() + "\"}"
        );
    }
}

// 配置
@Bean
public SecurityFilterChain securityFilterChain(HttpSecurity http) throws Exception {
    http.formLogin(form -> form
        .loginProcessingUrl("/api/login")
        .successHandler(loginSuccessHandler)
        .failureHandler(loginFailureHandler)
    );
    return http.build();
}
```

---

## JWT 认证

### JWT 工具类

```java
@Component
public class JwtUtils {
    
    @Value("${jwt.secret}")
    private String secret;
    
    @Value("${jwt.expiration:86400000}")  // 默认 24 小时
    private long expiration;
    
    private SecretKey getSignKey() {
        return Keys.hmacShaKeyFor(secret.getBytes());
    }
    
    // 生成 Token
    public String generateToken(String username, List<String> roles) {
        Map<String, Object> claims = new HashMap<>();
        claims.put("roles", roles);
        
        return Jwts.builder()
            .claims(claims)
            .subject(username)
            .issuedAt(new Date())
            .expiration(new Date(System.currentTimeMillis() + expiration))
            .signWith(getSignKey())
            .compact();
    }
    
    // 解析 Token
    public Claims parseToken(String token) {
        return Jwts.parser()
            .verifyWith(getSignKey())
            .build()
            .parseSignedClaims(token)
            .getPayload();
    }
    
    // 验证 Token
    public boolean validateToken(String token) {
        try {
            parseToken(token);
            return true;
        } catch (Exception e) {
            return false;
        }
    }
    
    // 从 Token 获取用户名
    public String getUsername(String token) {
        return parseToken(token).getSubject();
    }
}
```

### JWT 过滤器

```java
@Component
public class JwtAuthenticationFilter extends OncePerRequestFilter {
    
    @Autowired
    private JwtUtils jwtUtils;
    
    @Autowired
    private UserDetailsService userDetailsService;
    
    @Override
    protected void doFilterInternal(
            HttpServletRequest request, 
            HttpServletResponse response, 
            FilterChain filterChain) throws ServletException, IOException {
        
        // 获取 Token
        String token = getTokenFromRequest(request);
        
        if (token != null && jwtUtils.validateToken(token)) {
            String username = jwtUtils.getUsername(token);
            
            // 加载用户信息
            UserDetails userDetails = userDetailsService.loadUserByUsername(username);
            
            // 创建认证对象
            UsernamePasswordAuthenticationToken authentication = 
                new UsernamePasswordAuthenticationToken(
                    userDetails, null, userDetails.getAuthorities());
            
            authentication.setDetails(new WebAuthenticationDetailsSource().buildDetails(request));
            
            // 存入安全上下文
            SecurityContextHolder.getContext().setAuthentication(authentication);
        }
        
        filterChain.doFilter(request, response);
    }
    
    private String getTokenFromRequest(HttpServletRequest request) {
        String bearerToken = request.getHeader("Authorization");
        if (bearerToken != null && bearerToken.startsWith("Bearer ")) {
            return bearerToken.substring(7);
        }
        return null;
    }
}
```

### JWT 配置

```java
@Configuration
@EnableWebSecurity
public class JwtSecurityConfig {
    
    @Autowired
    private JwtAuthenticationFilter jwtAuthenticationFilter;
    
    @Bean
    public SecurityFilterChain securityFilterChain(HttpSecurity http) throws Exception {
        http
            // 禁用 CSRF（JWT 不需要）
            .csrf(csrf -> csrf.disable())
            // 禁用 Session
            .sessionManagement(session -> session
                .sessionCreationPolicy(SessionCreationPolicy.STATELESS)
            )
            // 授权配置
            .authorizeHttpRequests(auth -> auth
                .requestMatchers("/api/auth/**").permitAll()
                .anyRequest().authenticated()
            )
            // 添加 JWT 过滤器
            .addFilterBefore(jwtAuthenticationFilter, 
                UsernamePasswordAuthenticationFilter.class);
        
        return http.build();
    }
}
```

### 认证控制器

```java
@RestController
@RequestMapping("/api/auth")
@RequiredArgsConstructor
public class AuthController {
    
    private final AuthenticationManager authenticationManager;
    private final JwtUtils jwtUtils;
    private final UserDetailsService userDetailsService;
    
    @PostMapping("/login")
    public Result<LoginVO> login(@RequestBody LoginDTO dto) {
        // 认证
        Authentication authentication = authenticationManager.authenticate(
            new UsernamePasswordAuthenticationToken(dto.getUsername(), dto.getPassword())
        );
        
        // 生成 Token
        UserDetails userDetails = (UserDetails) authentication.getPrincipal();
        List<String> roles = userDetails.getAuthorities().stream()
            .map(GrantedAuthority::getAuthority)
            .collect(Collectors.toList());
        
        String token = jwtUtils.generateToken(userDetails.getUsername(), roles);
        
        return Result.success(new LoginVO(token, userDetails.getUsername()));
    }
    
    @GetMapping("/info")
    public Result<UserInfoVO> getUserInfo() {
        Authentication auth = SecurityContextHolder.getContext().getAuthentication();
        String username = auth.getName();
        // 返回用户信息
        return Result.success(new UserInfoVO(username));
    }
}
```

---

## 授权机制

### 基于角色的授权

```java
@Bean
public SecurityFilterChain securityFilterChain(HttpSecurity http) throws Exception {
    http.authorizeHttpRequests(auth -> auth
        // 角色检查
        .requestMatchers("/admin/**").hasRole("ADMIN")
        .requestMatchers("/user/**").hasAnyRole("USER", "ADMIN")
        
        // 权限检查
        .requestMatchers("/user/delete/**").hasAuthority("user:delete")
        .requestMatchers("/user/edit/**").hasAnyAuthority("user:edit", "user:admin")
    );
    return http.build();
}
```

### 方法级安全

```java
@Configuration
@EnableMethodSecurity  // 启用方法级安全
public class SecurityConfig { }

// 使用注解控制权限
@Service
public class UserService {
    
    @PreAuthorize("hasRole('ADMIN')")  // 方法执行前检查
    public void deleteUser(Long id) {
        // 删除用户
    }
    
    @PreAuthorize("hasAuthority('user:read') or #id == authentication.principal.id")
    public User getUser(Long id) {
        // 查看用户（管理员或本人）
        return userRepository.findById(id);
    }
    
    @PostAuthorize("returnObject.username == authentication.name")  // 方法执行后检查
    public User getProfile(Long id) {
        return userRepository.findById(id);
    }
    
    @PreAuthorize("isAuthenticated()")  // 只需登录
    public List<User> listUsers() {
        return userRepository.findAll();
    }
}
```

### 权限注解说明

| 注解 | 说明 |
|------|------|
| `@PreAuthorize` | 方法执行前检查 |
| `@PostAuthorize` | 方法执行后检查 |
| `@PreFilter` | 过滤方法参数 |
| `@PostFilter` | 过滤返回结果 |
| `@Secured` | 简单角色检查（已过时） |

---

## 常见配置

### CORS 配置

```java
@Bean
public SecurityFilterChain securityFilterChain(HttpSecurity http) throws Exception {
    http.cors(cors -> cors.configurationSource(corsConfigurationSource()));
    return http.build();
}

@Bean
public CorsConfigurationSource corsConfigurationSource() {
    CorsConfiguration configuration = new CorsConfiguration();
    configuration.setAllowedOrigins(Arrays.asList("http://localhost:3000"));
    configuration.setAllowedMethods(Arrays.asList("GET", "POST", "PUT", "DELETE"));
    configuration.setAllowedHeaders(Arrays.asList("*"));
    configuration.setAllowCredentials(true);
    configuration.setMaxAge(3600L);
    
    UrlBasedCorsConfigurationSource source = new UrlBasedCorsConfigurationSource();
    source.registerCorsConfiguration("/**", configuration);
    return source;
}
```

### CSRF 配置

```java
// 禁用 CSRF（适合前后端分离）
http.csrf(csrf -> csrf.disable());

// 自定义 CSRF 配置
http.csrf(csrf -> csrf
    .csrfTokenRepository(CookieCsrfTokenRepository.withHttpOnlyFalse())
    .ignoringRequestMatchers("/api/public/**")
);
```

### Session 配置

```java
http.sessionManagement(session -> session
    .sessionCreationPolicy(SessionCreationPolicy.STATELESS)  // 无状态
    // 或
    .sessionCreationPolicy(SessionCreationPolicy.IF_REQUIRED)
    .maximumSessions(1)                     // 单设备登录
    .maxSessionsPreventsLogin(true)         // 阻止新登录
    .expiredUrl("/login?expired")           // 会话过期跳转
);
```

---

## 安全最佳实践

### ⚠️ 密码安全

```java
// ✅ 使用 BCrypt 加密
@Bean
public PasswordEncoder passwordEncoder() {
    return new BCryptPasswordEncoder();
}

// ❌ 不要使用明文密码
user.setPassword(password);  // 危险！
```

### ⚠️ 敏感信息保护

```yaml
# 不要在配置文件中硬编码敏感信息
spring:
  datasource:
    password: ${DB_PASSWORD}  # 使用环境变量
```

### ⚠️ HTTPS 强制

```java
http.requiresChannel(channel -> channel
    .anyRequest().requiresSecure()
);
```

### ⚠️ 安全头配置

```java
http.headers(headers -> headers
    .frameOptions(HeadersConfigurer.FrameOptionsConfig::sameOrigin)
    .xssProtection(xss -> xss.disable())  // 前端处理
    .contentTypeOptions(Customizer.withDefaults())
);
```

### ⚠️ 输入验证

```java
// 始终验证用户输入
@PostMapping("/register")
public Result<Void> register(@RequestBody @Valid UserDTO dto) {
    // 使用 @Valid 进行参数校验
    userService.register(dto);
    return Result.success(null);
}
```

---

## 完整配置示例

```java
@Configuration
@EnableWebSecurity
@EnableMethodSecurity
@RequiredArgsConstructor
public class SecurityConfig {
    
    private final JwtAuthenticationFilter jwtAuthenticationFilter;
    private final LoginSuccessHandler loginSuccessHandler;
    private final LoginFailureHandler loginFailureHandler;
    
    @Bean
    public SecurityFilterChain securityFilterChain(HttpSecurity http) throws Exception {
        http
            // CSRF
            .csrf(csrf -> csrf.disable())
            
            // CORS
            .cors(Customizer.withDefaults())
            
            // Session
            .sessionManagement(session -> 
                session.sessionCreationPolicy(SessionCreationPolicy.STATELESS))
            
            // 授权规则
            .authorizeHttpRequests(auth -> auth
                .requestMatchers("/api/auth/**").permitAll()
                .requestMatchers("/api/public/**").permitAll()
                .requestMatchers("/api/admin/**").hasRole("ADMIN")
                .anyRequest().authenticated()
            )
            
            // 表单登录
            .formLogin(form -> form
                .loginProcessingUrl("/api/auth/login")
                .successHandler(loginSuccessHandler)
                .failureHandler(loginFailureHandler)
            )
            
            // JWT 过滤器
            .addFilterBefore(jwtAuthenticationFilter, 
                UsernamePasswordAuthenticationFilter.class)
            
            // 异常处理
            .exceptionHandling(exception -> exception
                .authenticationEntryPoint(new JwtAuthenticationEntryPoint())
            );
        
        return http.build();
    }
    
    @Bean
    public AuthenticationManager authenticationManager(
            UserDetailsService userDetailsService, 
            PasswordEncoder passwordEncoder) {
        DaoAuthenticationProvider provider = new DaoAuthenticationProvider();
        provider.setUserDetailsService(userDetailsService);
        provider.setPasswordEncoder(passwordEncoder);
        return new ProviderManager(provider);
    }
    
    @Bean
    public PasswordEncoder passwordEncoder() {
        return new BCryptPasswordEncoder();
    }
}
```

---

## 参考资料

- [Spring Security 官方文档](https://docs.spring.io/spring-security/reference/)
- [Spring Security 架构](https://spring.io/guides/topicals/spring-security-architecture)
- [JWT 官网](https://jwt.io/)
