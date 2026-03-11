# Spring MVC 与 RESTful API

> Spring MVC 是 Spring 框架的 Web 模块，提供了模型-视图-控制器架构的实现，是构建 Web 应用和 RESTful API 的核心框架。

##  MVC

### MVC 架构模式

MVC 是一种软件架构模式，将应用分为三个核心部分：

```
MVC 架构
│
├── Model（模型）
│   └── 数据和业务逻辑（如 User、Order 等实体类和 Service 层）
│
├── View（视图）
│   └── 用户界面展示（如 HTML 页面、JSON 响应）
│
└── Controller（控制器）
    └── 接收请求、调用业务逻辑、返回响应（协调 Model 和 View）
```

### Web 请求的处理流程

```
用户浏览器                    服务器
    │                          │
    │  1. 发送 HTTP 请求        │
    │ ──────────────────────→  │
    │                          │
    │                    2. Controller 接收请求
    │                          │
    │                    3. 调用 Service 处理业务
    │                          │
    │                    4. Service 操作数据库
    │                          │
    │                    5. 返回结果给 Controller
    │                          │
    │  6. 返回 HTTP 响应        │
    │ ←──────────────────────  │
    │                          │
```

### 传统 Servlet vs Spring MVC

```java
// ========== 传统 Servlet 方式 ==========
// 每个请求需要写一个 Servlet 类
public class UserServlet extends HttpServlet {
    @Override
    protected void doGet(HttpServletRequest req, HttpServletResponse resp) 
            throws ServletException, IOException {
        // 手动解析参数
        String id = req.getParameter("id");
        
        // 手动调用业务逻辑
        UserService userService = new UserServiceImpl();
        User user = userService.findById(Long.parseLong(id));
        
        // 手动设置响应
        resp.setContentType("application/json");
        resp.getWriter().write("{\"name\":\"" + user.getName() + "\"}");
    }
}

// 还需要在 web.xml 中配置 Servlet 映射
// <servlet>
//     <servlet-name>userServlet</servlet-name>
//     <servlet-class>com.example.UserServlet</servlet-class>
// </servlet>
// <servlet-mapping>
//     <servlet-name>userServlet</servlet-name>
//     <url-pattern>/user</url-pattern>
// </servlet-mapping>

// ========== Spring MVC 方式 ==========
// 一个方法处理一个请求，简洁明了
@RestController
@RequestMapping("/api/users")
public class UserController {
    
    @Autowired
    private UserService userService;
    
    @GetMapping("/{id}")
    public User getUser(@PathVariable Long id) {
        return userService.findById(id);  // 自动序列化为 JSON
    }
}
```

---

## 框架概述

Spring MVC 基于 Servlet API 构建，通过 DispatcherServlet 作为核心控制器。

### 核心组件

| 组件 | 说明 | 类比 |
|------|------|------|
| **DispatcherServlet** | 前端控制器，统一处理请求分发 | 总调度员 |
| **HandlerMapping** | 处理器映射，将请求映射到 Controller | 路由表 |
| **Controller** | 控制器，处理业务逻辑 | 具体办事员 |
| **ViewResolver** | 视图解析器，解析视图名称 | 模板引擎 |
| **HandlerAdapter** | 处理器适配器，执行 Controller 方法 | 执行器 |

### 请求处理流程

```
HTTP 请求
    ↓
DispatcherServlet（前端控制器）→ 收到请求
    ↓
HandlerMapping（处理器映射）→ 找到对应的 Controller 方法
    ↓
HandlerAdapter（处理器适配器）→ 执行 Controller 方法
    ↓
Controller（控制器）→ 调用 Service 处理业务
    ↓
返回 ModelAndView / 响应数据
    ↓
ViewResolver（视图解析器）→ 如果是视图渲染（REST API 跳过）
    ↓
HTTP 响应 → 返回给客户端
```

---

## 控制器详解

### @Controller 与 @RestController

```java
// ========== @Controller：传统 MVC，返回视图 ==========
// 用于服务端渲染的 Web 应用（如 JSP、Thymeleaf）
@Controller
public class PageController {
    
    @GetMapping("/home")
    public String home(Model model) {
        // 向视图传递数据
        model.addAttribute("message", "Hello");
        model.addAttribute("users", userService.findAll());
        
        // 返回视图名称（由 ViewResolver 解析）
        // 如果配置了 prefix=/WEB-INF/views/, suffix=.jsp
        // 实际路径：/WEB-INF/views/home.jsp
        return "home";
    }
    
    // 如果需要返回 JSON，需要加 @ResponseBody
    @GetMapping("/api/user")
    @ResponseBody
    public User getUser() {
        return userService.findById(1L);
    }
}

// ========== @RestController：RESTful API，返回 JSON ==========
// @RestController = @Controller + @ResponseBody
// 所有方法返回值都会自动序列化为 JSON
@RestController
@RequestMapping("/api/users")  // 类级别路径前缀
public class UserController {
    
    @Autowired
    private UserService userService;
    
    // GET /api/users/1
    @GetMapping("/{id}")
    public User getUser(@PathVariable Long id) {
        return userService.findById(id);
        // 自动序列化为 JSON：{"id":1,"name":"张三","age":25}
    }
}
```

### 请求映射注解

```java
@RestController
@RequestMapping("/api/users")  // 类级别路径前缀
public class UserController {
    
    // ========== GET 请求：查询资源 ==========
    
    // GET /api/users/1
    @GetMapping("/{id}")
    public User getById(@PathVariable Long id) { 
        return userService.findById(id); 
    }
    
    // GET /api/users?name=张三
    @GetMapping
    public List<User> list(@RequestParam(required = false) String name) { 
        return userService.findAll(name); 
    }
    
    // ========== POST 请求：创建资源 ==========
    
    // POST /api/users
    // Body: {"name":"张三","age":25}
    @PostMapping
    public User create(@RequestBody UserDTO dto) { 
        return userService.create(dto); 
    }
    
    // ========== PUT 请求：全量更新资源 ==========
    
    // PUT /api/users/1
    // Body: {"name":"李四","age":30}
    @PutMapping("/{id}")
    public User update(@PathVariable Long id, @RequestBody UserDTO dto) { 
        return userService.update(id, dto); 
    }
    
    // ========== PATCH 请求：部分更新资源 ==========
    
    // PATCH /api/users/1
    // Body: {"age":26}
    @PatchMapping("/{id}")
    public User partialUpdate(
            @PathVariable Long id, 
            @RequestBody Map<String, Object> updates) { 
        return userService.partialUpdate(id, updates); 
    }
    
    // ========== DELETE 请求：删除资源 ==========
    
    // DELETE /api/users/1
    @DeleteMapping("/{id}")
    public void delete(@PathVariable Long id) { 
        userService.delete(id); 
    }
    
    // ========== 多路径映射 ==========
    
    @GetMapping({"/detail/{id}", "/info/{id}"})
    public User detail(@PathVariable Long id) { 
        return userService.findById(id); 
    }
}
```

### HTTP 方法语义

| 方法 | 语义 | 幂等性 | 安全性 | 使用场景 |
|------|------|:------:|:------:|----------|
| GET | 获取资源 | ✓ | ✓ | 查询数据 |
| POST | 创建资源 | ✗ | ✗ | 新增数据 |
| PUT | 全量更新 | ✓ | ✗ | 替换整个资源 |
| PATCH | 部分更新 | ✗ | ✗ | 修改部分字段 |
| DELETE | 删除资源 | ✓ | ✗ | 删除数据 |

**幂等性**：多次相同请求，结果相同
**安全性**：不会修改服务器数据

---

## 参数绑定

Spring MVC 自动将请求参数绑定到方法参数。

### 路径变量 @PathVariable

```java
// 从 URL 路径中获取参数

// GET /api/users/1
@GetMapping("/users/{id}")
public User getUser(@PathVariable Long id) { 
    // id = 1
}

// GET /api/users/1/orders/100
@GetMapping("/users/{userId}/orders/{orderId}")
public Order getOrder(
        @PathVariable Long userId,
        @PathVariable Long orderId) {
    // userId = 1, orderId = 100
}

// 参数名不一致时指定名称
@GetMapping("/users/{id}")
public User getUser(@PathVariable("id") Long userId) {
    // userId = URL 中的 id 值
}

// 可选的路径变量（Java 8+，需要设置 required=false）
@GetMapping("/users/{id}")
public User getUser(@PathVariable(required = false) Long id) {
    // id 可能为 null
}
```

### 请求参数 @RequestParam

```java
// 从 URL 查询字符串获取参数

// GET /api/users?keyword=张三
@GetMapping("/search")
public List<User> search(@RequestParam String keyword) { 
    // keyword = "张三"
}

// GET /api/users?page=1&size=10
@GetMapping
public Page<User> list(
        @RequestParam(defaultValue = "1") int page,      // 默认值
        @RequestParam(defaultValue = "10") int size) {
    // page = 1, size = 10
}

// 可选参数
@GetMapping("/search")
public List<User> search(
        @RequestParam(required = false) String keyword,  // 可选
        @RequestParam(defaultValue = "10") int limit) {
    // keyword 可能为 null
}

// 参数名不一致时指定名称
@GetMapping("/search")
public List<User> search(@RequestParam("q") String keyword) {
    // keyword = URL 中 q 参数的值
}

// 多值参数
// GET /api/users?tags=java&tags=spring
@GetMapping("/filter")
public List<User> filter(@RequestParam List<String> tags) {
    // tags = ["java", "spring"]
}

// 使用 Map 接收所有参数
@GetMapping("/search")
public List<User> search(@RequestParam Map<String, String> params) {
    // params 包含所有查询参数
}
```

### 请求体 @RequestBody

```java
// 从请求体获取 JSON 数据

// POST /api/users
// Body: {"name":"张三","age":25}
@PostMapping
public User create(@RequestBody UserDTO dto) {
    // dto.getName() = "张三"
    // dto.getAge() = 25
}

// 接收 Map（灵活但类型不安全）
@PostMapping("/config")
public void updateConfig(@RequestBody Map<String, Object> config) {
    // config = {"key1": "value1", "key2": 123}
}

// 接收 List
// Body: [{"name":"张三"},{"name":"李四"}]
@PostMapping("/batch")
public List<User> batchCreate(@RequestBody List<UserDTO> dtos) {
    // dtos 是多个 UserDTO 对象
}
```

### 请求头 @RequestHeader

```java
@GetMapping("/info")
public String getInfo(
        @RequestHeader("Authorization") String auth,           // 必需
        @RequestHeader(value = "User-Agent", required = false) String userAgent,  // 可选
        @RequestHeader(value = "X-Token", defaultValue = "") String token) {  // 默认值
    
    return "Auth: " + auth;
}

// 获取所有请求头
@GetMapping("/headers")
public Map<String, String> headers(@RequestHeader Map<String, String> headers) {
    return headers;
}
```

### Cookie @CookieValue

```java
@GetMapping("/session")
public String getSession(
        @CookieValue("JSESSIONID") String sessionId,
        @CookieValue(value = "token", required = false) String token) {
    return "Session ID: " + sessionId;
}
```

### 实体对象绑定

```java
// 自动将请求参数绑定到对象属性
// GET /api/query?name=张三&age=25&status=active
@GetMapping("/query")
public List<User> query(UserQuery query) {
    // query.getName() = "张三"
    // query.getAge() = 25
    // query.getStatus() = "active"
    return userService.query(query);
}

@Data
public class UserQuery {
    private String name;
    private Integer age;
    private String status;
}
```

---

## 响应处理

### 返回 JSON

```java
@RestController
public class UserController {
    
    // 返回对象，自动序列化为 JSON
    @GetMapping("/users/{id}")
    public User getUser(@PathVariable Long id) {
        return userService.findById(id);
        // 响应：{"id":1,"name":"张三","age":25}
    }
    
    // 返回集合
    @GetMapping("/users")
    public List<User> list() {
        return userService.findAll();
        // 响应：[{"id":1,...}, {"id":2,...}]
    }
    
    // 返回 Map
    @GetMapping("/stats")
    public Map<String, Object> stats() {
        Map<String, Object> map = new HashMap<>();
        map.put("total", 100);
        map.put("active", 80);
        return map;
        // 响应：{"total":100,"active":80}
    }
}
```

### ResponseEntity 灵活响应

```java
// ResponseEntity 可以完全控制响应
// 包括状态码、响应头、响应体

@GetMapping("/users/{id}")
public ResponseEntity<User> getUser(@PathVariable Long id) {
    User user = userService.findById(id);
    
    if (user == null) {
        return ResponseEntity.notFound().build();  // 404
    }
    return ResponseEntity.ok(user);  // 200 + JSON
}

@PostMapping("/users")
public ResponseEntity<User> create(
        @RequestBody UserDTO dto, 
        UriComponentsBuilder uriBuilder) {  // 构建 URL
    
    User user = userService.create(dto);
    
    // 构建 Location 响应头
    URI location = uriBuilder
        .path("/api/users/{id}")
        .buildAndExpand(user.getId())
        .toUri();
    
    return ResponseEntity
        .created(location)    // 201 Created
        .header("X-Custom-Header", "value")
        .body(user);
}

// 常用状态码
ResponseEntity.ok(body);                      // 200 OK
ResponseEntity.created(location).build();    // 201 Created
ResponseEntity.noContent().build();          // 204 No Content
ResponseEntity.badRequest().build();         // 400 Bad Request
ResponseEntity.notFound().build();           // 404 Not Found
ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();  // 500
```

### 文件下载

```java
@GetMapping("/download/{filename}")
public ResponseEntity<Resource> download(@PathVariable String filename) throws IOException {
    // 加载文件
    Path path = Paths.get("/uploads/" + filename);
    Resource resource = new UrlResource(path.toUri());
    
    if (!resource.exists()) {
        return ResponseEntity.notFound().build();
    }
    
    return ResponseEntity.ok()
        .contentType(MediaType.APPLICATION_OCTET_STREAM)
        .header(HttpHeaders.CONTENT_DISPOSITION, 
                "attachment; filename=\"" + URLEncoder.encode(filename, "UTF-8") + "\"")
        .body(resource);
}
```

### 文件上传

```java
@PostMapping("/upload")
public String upload(@RequestParam("file") MultipartFile file) throws IOException {
    if (file.isEmpty()) {
        return "文件为空";
    }
    
    // 获取文件信息
    String originalName = file.getOriginalFilename();  // 原始文件名
    String contentType = file.getContentType();        // 文件类型
    long size = file.getSize();                        // 文件大小
    
    // 保存文件
    String filename = UUID.randomUUID() + "_" + originalName;
    Path path = Paths.get("/uploads/" + filename);
    Files.copy(file.getInputStream(), path);
    
    return "上传成功: " + filename;
}

// 多文件上传
@PostMapping("/uploads")
public String uploads(@RequestParam("files") MultipartFile[] files) throws IOException {
    for (MultipartFile file : files) {
        if (!file.isEmpty()) {
            // 处理每个文件
        }
    }
    return "上传成功";
}
```

---

## RESTful API 设计规范

### URL 设计规范

```
# ✅ 推荐的设计

# 资源用名词（复数形式）
GET    /api/users           # 获取用户列表
GET    /api/users/1         # 获取单个用户
POST   /api/users           # 创建用户
PUT    /api/users/1         # 全量更新用户
PATCH  /api/users/1         # 部分更新用户
DELETE /api/users/1         # 删除用户

# 子资源
GET    /api/users/1/orders           # 获取用户的订单列表
GET    /api/users/1/orders/2         # 获取用户的特定订单

# 过滤、排序、分页
GET    /api/users?status=active          # 过滤
GET    /api/users?sort=name&order=asc    # 排序
GET    /api/users?page=1&size=10         # 分页

# ❌ 避免的设计

# 不要在 URL 中使用动词
GET    /api/getUsers        # ❌ 应该用 GET /api/users
POST   /api/createUser      # ❌ 应该用 POST /api/users
DELETE /api/deleteUser?id=1 # ❌ 应该用 DELETE /api/users/1

# 不要使用嵌套过深的资源
GET    /api/users/1/orders/2/items/3/products/4  # ❌ 太深了
```

### 状态码使用规范

| 状态码 | 含义 | 使用场景 |
|--------|------|----------|
| 200 | OK | 成功返回数据 |
| 201 | Created | 成功创建资源 |
| 204 | No Content | 成功但无返回数据（如删除成功） |
| 400 | Bad Request | 请求参数错误 |
| 401 | Unauthorized | 未认证 |
| 403 | Forbidden | 已认证但无权限 |
| 404 | Not Found | 资源不存在 |
| 409 | Conflict | 资源冲突（如用户名已存在） |
| 422 | Unprocessable Entity | 参数校验失败 |
| 500 | Internal Server Error | 服务器错误 |

```java
// 示例
@PostMapping("/users")
public ResponseEntity<User> create(@RequestBody UserDTO dto) {
    User user = userService.create(dto);
    return ResponseEntity.status(HttpStatus.CREATED).body(user);  // 201
}

@DeleteMapping("/users/{id}")
public ResponseEntity<Void> delete(@PathVariable Long id) {
    userService.delete(id);
    return ResponseEntity.noContent().build();  // 204
}
```

---

## 拦截器

拦截器用于在请求到达 Controller 前后进行处理。

### 实现拦截器

```java
@Component
public class AuthInterceptor implements HandlerInterceptor {
    
    /**
     * 前置处理：Controller 方法执行前
     * 返回 true 继续执行，返回 false 终止请求
     */
    @Override
    public boolean preHandle(
            HttpServletRequest request, 
            HttpServletResponse response, 
            Object handler) throws Exception {
        
        String token = request.getHeader("Authorization");
        
        if (token == null || !validateToken(token)) {
            // 未认证，返回 401
            response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
            response.setContentType("application/json;charset=UTF-8");
            response.getWriter().write("{\"code\":401,\"message\":\"未授权\"}");
            return false;  // 拦截请求
        }
        
        // 将用户信息存入 request
        request.setAttribute("userId", getUserIdFromToken(token));
        return true;  // 放行
    }
    
    /**
     * 后置处理：Controller 方法执行后，视图渲染前
     */
    @Override
    public void postHandle(
            HttpServletRequest request, 
            HttpServletResponse response, 
            Object handler, 
            ModelAndView modelAndView) throws Exception {
        // 可以修改 ModelAndView
        if (modelAndView != null) {
            modelAndView.addObject("commonData", "公共数据");
        }
    }
    
    /**
     * 完成处理：视图渲染后（无论是否异常）
     */
    @Override
    public void afterCompletion(
            HttpServletRequest request, 
            HttpServletResponse response, 
            Object handler, 
            Exception ex) throws Exception {
        // 资源清理、日志记录
        long startTime = (Long) request.getAttribute("startTime");
        long elapsed = System.currentTimeMillis() - startTime;
        log.info("请求 {} 耗时: {}ms", request.getRequestURI(), elapsed);
    }
    
    private boolean validateToken(String token) {
        // Token 验证逻辑
        return true;
    }
    
    private Long getUserIdFromToken(String token) {
        return 1L;
    }
}
```

### 注册拦截器

```java
@Configuration
public class WebConfig implements WebMvcConfigurer {
    
    @Autowired
    private AuthInterceptor authInterceptor;
    
    @Override
    public void addInterceptors(InterceptorRegistry registry) {
        registry.addInterceptor(authInterceptor)
            .addPathPatterns("/api/**")          // 拦截路径
            .excludePathPatterns(                // 排除路径
                "/api/auth/login",
                "/api/auth/register",
                "/api/public/**",
                "/error"
            )
            .order(1);  // 拦截器顺序，数字越小越先执行
        
        // 可以添加多个拦截器
        // registry.addInterceptor(loggingInterceptor).order(2);
    }
}
```

### 拦截器 vs 过滤器

| 特性 | 拦截器（Interceptor） | 过滤器（Filter） |
|------|----------------------|------------------|
| 所属 | Spring MVC | Servlet 容器 |
| 拦截范围 | Controller 方法 | 所有请求 |
| 访问 Spring Bean | 可以 | 需要配置 |
| 执行顺序 | 在过滤器之后 | 在拦截器之前 |

```java
// 过滤器示例
@Component
@WebFilter(urlPatterns = "/*")
public class LogFilter implements Filter {
    
    @Override
    public void doFilter(ServletRequest request, ServletResponse response, FilterChain chain) 
            throws IOException, ServletException {
        
        HttpServletRequest req = (HttpServletRequest) request;
        long startTime = System.currentTimeMillis();
        
        chain.doFilter(request, response);  // 继续执行
        
        long elapsed = System.currentTimeMillis() - startTime;
        log.info("{} {} - {}ms", req.getMethod(), req.getRequestURI(), elapsed);
    }
}
```

---

## 跨域处理

### 为什么会有跨域问题？

```
浏览器安全策略：同源策略
- 协议、域名、端口必须完全相同
- 不同源的请求会被浏览器拦截

示例：
前端：http://localhost:3000
后端：http://localhost:8080

→ 端口不同，属于跨域请求
→ 浏览器会拦截响应
```

### @CrossOrigin 注解

```java
// 类级别跨域配置
@RestController
@RequestMapping("/api/users")
@CrossOrigin(
    origins = "http://localhost:3000",  // 允许的源
    methods = {RequestMethod.GET, RequestMethod.POST},
    allowedHeaders = "*",
    allowCredentials = "true",
    maxAge = 3600  // 预检请求缓存时间（秒）
)
public class UserController { }

// 方法级别
@GetMapping("/{id}")
@CrossOrigin(origins = "*")  // 允许所有源
public User getUser(@PathVariable Long id) { }
```

### 全局跨域配置

```java
@Configuration
public class WebConfig implements WebMvcConfigurer {
    
    @Override
    public void addCorsMappings(CorsRegistry registry) {
        registry.addMapping("/api/**")  // 拦截路径
            .allowedOriginPatterns("http://localhost:*", "https://example.com")
            .allowedMethods("GET", "POST", "PUT", "DELETE", "OPTIONS")
            .allowedHeaders("*")
            .allowCredentials(true)
            .maxAge(3600);
    }
}
```

---

## 异常处理

### @ExceptionHandler 局部异常处理

```java
@RestController
public class UserController {
    
    // 只处理当前 Controller 中的异常
    @ExceptionHandler(UserNotFoundException.class)
    public ResponseEntity<Result<Void>> handleUserNotFound(UserNotFoundException e) {
        return ResponseEntity.status(HttpStatus.NOT_FOUND)
            .body(Result.error(404, e.getMessage()));
    }
    
    @ExceptionHandler(IllegalArgumentException.class)
    public ResponseEntity<Result<Void>> handleIllegalArgument(IllegalArgumentException e) {
        return ResponseEntity.badRequest()
            .body(Result.error(400, e.getMessage()));
    }
}
```

### @ControllerAdvice 全局异常处理

```java
// 全局异常处理器
@RestControllerAdvice  // 对所有 @RestController 生效
@Slf4j
public class GlobalExceptionHandler {
    
    // 处理自定义业务异常
    @ExceptionHandler(BusinessException.class)
    public ResponseEntity<Result<Void>> handleBusinessException(BusinessException e) {
        log.warn("业务异常: {}", e.getMessage());
        return ResponseEntity.badRequest()
            .body(Result.error(e.getCode(), e.getMessage()));
    }
    
    // 处理资源不存在
    @ExceptionHandler(EntityNotFoundException.class)
    public ResponseEntity<Result<Void>> handleNotFound(EntityNotFoundException e) {
        return ResponseEntity.status(HttpStatus.NOT_FOUND)
            .body(Result.error(404, "资源不存在"));
    }
    
    // 处理参数校验异常
    @ExceptionHandler(MethodArgumentNotValidException.class)
    public ResponseEntity<Result<Void>> handleValidation(MethodArgumentNotValidException e) {
        String message = e.getBindingResult()
            .getFieldErrors()
            .stream()
            .map(FieldError::getDefaultMessage)
            .collect(Collectors.joining(", "));
        return ResponseEntity.badRequest()
            .body(Result.error(400, "参数校验失败: " + message));
    }
    
    // 处理参数绑定异常
    @ExceptionHandler(BindException.class)
    public ResponseEntity<Result<Void>> handleBindException(BindException e) {
        String message = e.getBindingResult()
            .getFieldErrors()
            .stream()
            .map(FieldError::getDefaultMessage)
            .collect(Collectors.joining(", "));
        return ResponseEntity.badRequest()
            .body(Result.error(400, "参数绑定失败: " + message));
    }
    
    // 处理请求方法不支持
    @ExceptionHandler(HttpRequestMethodNotSupportedException.class)
    public ResponseEntity<Result<Void>> handleMethodNotSupported(HttpRequestMethodNotSupportedException e) {
        return ResponseEntity.status(HttpStatus.METHOD_NOT_ALLOWED)
            .body(Result.error(405, "不支持的请求方法: " + e.getMethod()));
    }
    
    // 处理所有未捕获异常
    @ExceptionHandler(Exception.class)
    public ResponseEntity<Result<Void>> handleException(Exception e) {
        log.error("系统异常", e);
        return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
            .body(Result.error(500, "系统繁忙，请稍后重试"));
    }
}
```

---

## 完整示例

### Controller 层

```java
@RestController
@RequestMapping("/api/users")
@RequiredArgsConstructor
@Tag(name = "用户管理", description = "用户相关接口")  // OpenAPI 文档注解
public class UserController {
    
    private final UserService userService;
    
    @GetMapping
    @Operation(summary = "获取用户列表")
    public Result<Page<UserVO>> list(
            @RequestParam(defaultValue = "1") int page,
            @RequestParam(defaultValue = "10") int size,
            @RequestParam(required = false) String keyword) {
        return Result.success(userService.list(page, size, keyword));
    }
    
    @GetMapping("/{id}")
    @Operation(summary = "获取用户详情")
    public Result<UserVO> getById(@PathVariable Long id) {
        return Result.success(userService.getById(id));
    }
    
    @PostMapping
    @Operation(summary = "创建用户")
    public Result<UserVO> create(@RequestBody @Valid UserCreateDTO dto) {
        return Result.success(userService.create(dto));
    }
    
    @PutMapping("/{id}")
    @Operation(summary = "更新用户")
    public Result<UserVO> update(
            @PathVariable Long id, 
            @RequestBody @Valid UserUpdateDTO dto) {
        return Result.success(userService.update(id, dto));
    }
    
    @DeleteMapping("/{id}")
    @Operation(summary = "删除用户")
    public Result<Void> delete(@PathVariable Long id) {
        userService.delete(id);
        return Result.success();
    }
}
```

---

## 小结

| 概念 | 说明 |
|------|------|
| **@RestController** | RESTful 控制器，返回 JSON |
| **@RequestMapping** | 请求映射，支持路径、方法、参数等条件 |
| **@PathVariable** | 绑定路径变量 |
| **@RequestParam** | 绑定请求参数 |
| **@RequestBody** | 绑定请求体 JSON |
| **ResponseEntity** | 灵活控制响应 |
| **拦截器** | 请求预处理和后处理 |
| **跨域处理** | @CrossOrigin 或全局配置 |
| **异常处理** | @ExceptionHandler + @RestControllerAdvice |

### RESTful API 设计要点

1. **URL 用名词**：`/users` 而不是 `/getUsers`
2. **HTTP 方法表达操作**：GET 查询、POST 创建、PUT 更新、DELETE 删除
3. **正确使用状态码**：200 成功、201 创建、400 参数错误、404 不存在
4. **统一响应格式**：code + message + data
5. **全局异常处理**：统一处理异常，返回友好提示