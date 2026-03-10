# Spring MVC 与 RESTful API

> Spring MVC 是 Spring 框架的 Web 模块，提供了模型-视图-控制器架构的实现，是构建 Web 应用和 RESTful API 的核心框架。

## 框架概述

Spring MVC 基于 Servlet API 构建，通过 DispatcherServlet 作为核心控制器，将请求分发给相应的处理器，实现了 Web 层的松耦合设计。

### 🎯 核心组件

| 组件 | 说明 |
|------|------|
| **DispatcherServlet** | 前端控制器，统一处理请求分发 |
| **HandlerMapping** | 处理器映射，将请求映射到 Controller |
| **Controller** | 控制器，处理业务逻辑 |
| **ViewResolver** | 视图解析器，解析视图名称到实际视图 |
| **HandlerAdapter** | 处理器适配器，执行 Controller 方法 |

### 请求处理流程

```
HTTP 请求
    ↓
DispatcherServlet（前端控制器）
    ↓
HandlerMapping（查找 Controller）
    ↓
HandlerAdapter（执行方法）
    ↓
Controller（业务处理）
    ↓
返回 ModelAndView / 响应数据
    ↓
ViewResolver（视图解析，非 REST API 跳过）
    ↓
HTTP 响应
```

---

## 控制器详解

### @Controller 与 @RestController

```java
// 传统 MVC 控制器，返回视图名称
@Controller
public class PageController {
    
    @GetMapping("/home")
    public String home(Model model) {
        model.addAttribute("message", "Hello");
        return "home";  // 返回视图名称，由 ViewResolver 解析
    }
}

// RESTful 控制器，返回 JSON 数据
@RestController  // = @Controller + @ResponseBody
@RequestMapping("/api/users")
public class UserController {
    
    @GetMapping("/{id}")
    public User getUser(@PathVariable Long id) {
        return userService.findById(id);  // 自动序列化为 JSON
    }
}
```

### 请求映射注解

```java
@RestController
@RequestMapping("/api/users")  // 类级别路径前缀
public class UserController {
    
    // GET 请求
    @GetMapping("/{id}")
    public User getById(@PathVariable Long id) { }
    
    // GET 请求 - 查询参数
    @GetMapping
    public List<User> list(@RequestParam(required = false) String name) { }
    
    // POST 请求
    @PostMapping
    public User create(@RequestBody UserDTO dto) { }
    
    // PUT 请求 - 全量更新
    @PutMapping("/{id}")
    public User update(@PathVariable Long id, @RequestBody UserDTO dto) { }
    
    // PATCH 请求 - 部分更新
    @PatchMapping("/{id}")
    public User partialUpdate(@PathVariable Long id, @RequestBody Map<String, Object> updates) { }
    
    // DELETE 请求
    @DeleteMapping("/{id}")
    public void delete(@PathVariable Long id) { }
    
    // 多路径映射
    @GetMapping({"/detail/{id}", "/info/{id}"})
    public User detail(@PathVariable Long id) { }
}
```

---

## 参数绑定

### 路径变量 @PathVariable

```java
@GetMapping("/users/{id}")
public User getUser(@PathVariable Long id) { }

// 多个路径变量
@GetMapping("/users/{userId}/orders/{orderId}")
public Order getOrder(
        @PathVariable Long userId,
        @PathVariable Long orderId) { }

// 名称不一致时指定名称
@GetMapping("/users/{id}")
public User getUser(@PathVariable("id") Long userId) { }
```

### 请求参数 @RequestParam

```java
// 必填参数
@GetMapping("/search")
public List<User> search(@RequestParam String keyword) { }

// 可选参数 + 默认值
@GetMapping("/list")
public Page<User> list(
        @RequestParam(defaultValue = "1") int page,
        @RequestParam(defaultValue = "10") int size) { }

// 参数名不一致
@GetMapping("/search")
public List<User> search(@RequestParam("q") String keyword) { }

// 多值参数
@GetMapping("/filter")
public List<User> filter(@RequestParam List<String> tags) { }
// URL: /filter?tags=java&tags=spring
```

### 请求体 @RequestBody

```java
@PostMapping
public User create(@RequestBody UserDTO dto) { }

// 接收 Map
@PostMapping("/config")
public void updateConfig(@RequestBody Map<String, Object> config) { }
```

### 请求头 @RequestHeader

```java
@GetMapping("/info")
public String getInfo(
        @RequestHeader("Authorization") String auth,
        @RequestHeader(value = "User-Agent", required = false) String userAgent) {
    return "Auth: " + auth;
}
```

### Cookie @CookieValue

```java
@GetMapping("/session")
public String getSession(@CookieValue("JSESSIONID") String sessionId) {
    return "Session ID: " + sessionId;
}
```

### 实体对象绑定

```java
@GetMapping("/query")
public List<User> query(UserQuery query) {
    // 自动绑定查询参数到对象属性
    // URL: /query?name=John&age=20&status=active
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
    
    @GetMapping("/users/{id}")
    public User getUser(@PathVariable Long id) {
        return userService.findById(id);  // 自动序列化为 JSON
    }
    
    // 自定义响应
    @PostMapping("/users")
    public ResponseEntity<User> create(@RequestBody UserDTO dto) {
        User user = userService.create(dto);
        return ResponseEntity
            .status(HttpStatus.CREATED)
            .header("X-Custom-Header", "value")
            .body(user);
    }
}
```

### ResponseEntity 灵活响应

```java
@GetMapping("/users/{id}")
public ResponseEntity<User> getUser(@PathVariable Long id) {
    User user = userService.findById(id);
    if (user == null) {
        return ResponseEntity.notFound().build();
    }
    return ResponseEntity.ok(user);
}

@PostMapping("/users")
public ResponseEntity<User> create(@RequestBody UserDTO dto, UriComponentsBuilder uriBuilder) {
    User user = userService.create(dto);
    
    // 构建 Location 头
    URI location = uriBuilder
        .path("/api/users/{id}")
        .buildAndExpand(user.getId())
        .toUri();
    
    return ResponseEntity
        .created(location)
        .body(user);
}
```

### 文件下载

```java
@GetMapping("/download/{filename}")
public ResponseEntity<Resource> download(@PathVariable String filename) throws IOException {
    Path path = Paths.get("/uploads/" + filename);
    Resource resource = new UrlResource(path.toUri());
    
    return ResponseEntity.ok()
        .contentType(MediaType.APPLICATION_OCTET_STREAM)
        .header(HttpHeaders.CONTENT_DISPOSITION, 
                "attachment; filename=\"" + resource.getFilename() + "\"")
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
    
    String filename = UUID.randomUUID() + "_" + file.getOriginalFilename();
    Path path = Paths.get("/uploads/" + filename);
    Files.copy(file.getInputStream(), path);
    
    return "上传成功: " + filename;
}

// 多文件上传
@PostMapping("/uploads")
public String uploads(@RequestParam("files") MultipartFile[] files) {
    for (MultipartFile file : files) {
        // 处理每个文件
    }
    return "上传成功";
}
```

---

## RESTful API 设计规范

### HTTP 方法语义

| 方法 | 语义 | 幂等性 | 安全性 |
|------|------|:------:|:------:|
| GET | 查询资源 | ✅ | ✅ |
| POST | 创建资源 | ❌ | ❌ |
| PUT | 全量更新资源 | ✅ | ❌ |
| PATCH | 部分更新资源 | ❌ | ❌ |
| DELETE | 删除资源 | ✅ | ❌ |

### URL 设计规范

```
# ✅ 推荐
GET    /api/users          # 获取用户列表
GET    /api/users/1        # 获取单个用户
POST   /api/users          # 创建用户
PUT    /api/users/1        # 全量更新用户
PATCH  /api/users/1        # 部分更新用户
DELETE /api/users/1        # 删除用户

# 子资源
GET    /api/users/1/orders          # 获取用户的订单列表
GET    /api/users/1/orders/2        # 获取用户的特定订单

# ❌ 避免
GET    /api/getUsers        # 动词应使用 HTTP 方法表示
POST   /api/createUser
DELETE /api/deleteUser?id=1
```

### 状态码使用

```java
// 成功状态码
return ResponseEntity.ok(user);                    // 200 OK
return ResponseEntity.created(location).build();   // 201 Created
return ResponseEntity.noContent().build();         // 204 No Content

// 客户端错误
return ResponseEntity.badRequest().build();        // 400 Bad Request
return ResponseEntity.notFound().build();          // 404 Not Found
return ResponseEntity.status(HttpStatus.CONFLICT).build();  // 409 Conflict

// 服务器错误
return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();  // 500
```

---

## 拦截器

### 实现拦截器

```java
@Component
public class AuthInterceptor implements HandlerInterceptor {
    
    // 前置处理：Controller 方法执行前
    @Override
    public boolean preHandle(HttpServletRequest request, 
                            HttpServletResponse response, 
                            Object handler) throws Exception {
        String token = request.getHeader("Authorization");
        if (token == null || !validateToken(token)) {
            response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
            return false;  // 拦截请求
        }
        return true;  // 放行
    }
    
    // 后置处理：Controller 方法执行后，视图渲染前
    @Override
    public void postHandle(HttpServletRequest request, 
                          HttpServletResponse response, 
                          Object handler, 
                          ModelAndView modelAndView) throws Exception {
        // 可修改 ModelAndView
    }
    
    // 完成处理：视图渲染后
    @Override
    public void afterCompletion(HttpServletRequest request, 
                               HttpServletResponse response, 
                               Object handler, 
                               Exception ex) throws Exception {
        // 资源清理
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
                "/api/public/**"
            );
    }
}
```

---

## 跨域处理

### @CrossOrigin 注解

```java
@RestController
@RequestMapping("/api/users")
@CrossOrigin(
    origins = "http://localhost:3000",  // 允许的源
    methods = {RequestMethod.GET, RequestMethod.POST},
    maxAge = 3600  // 预检请求缓存时间
)
public class UserController { }

// 方法级别
@GetMapping("/{id}")
@CrossOrigin(origins = "*")
public User getUser(@PathVariable Long id) { }
```

### 全局跨域配置

```java
@Configuration
public class WebConfig implements WebMvcConfigurer {
    
    @Override
    public void addCorsMappings(CorsRegistry registry) {
        registry.addMapping("/api/**")
            .allowedOrigins("http://localhost:3000", "https://example.com")
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
    
    @ExceptionHandler(UserNotFoundException.class)
    public ResponseEntity<String> handleUserNotFound(UserNotFoundException e) {
        return ResponseEntity.status(HttpStatus.NOT_FOUND).body(e.getMessage());
    }
}
```

### @ControllerAdvice 全局异常处理

```java
@RestControllerAdvice
public class GlobalExceptionHandler {
    
    // 处理自定义异常
    @ExceptionHandler(BusinessException.class)
    public ResponseEntity<Result<Void>> handleBusinessException(BusinessException e) {
        return ResponseEntity.badRequest()
            .body(Result.error(e.getMessage()));
    }
    
    // 处理资源不存在
    @ExceptionHandler(EntityNotFoundException.class)
    public ResponseEntity<Result<Void>> handleNotFound(EntityNotFoundException e) {
        return ResponseEntity.status(HttpStatus.NOT_FOUND)
            .body(Result.error("资源不存在"));
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
            .body(Result.error("参数校验失败: " + message));
    }
    
    // 处理所有异常
    @ExceptionHandler(Exception.class)
    public ResponseEntity<Result<Void>> handleException(Exception e) {
        log.error("系统异常", e);
        return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
            .body(Result.error("系统繁忙，请稍后重试"));
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
@Tag(name = "用户管理", description = "用户相关接口")
public class UserController {
    
    private final UserService userService;
    
    @GetMapping
    @Operation(summary = "获取用户列表")
    public Result<Page<UserVO>> list(
            @RequestParam(defaultValue = "1") int page,
            @RequestParam(defaultValue = "10") int size) {
        return Result.success(userService.list(page, size));
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
        return Result.success(null);
    }
}
```

---

## 参考资料

- [Spring MVC 官方文档](https://docs.spring.io/spring-framework/reference/web/webmvc.html)
- [RESTful API 设计指南](https://restfulapi.net/)
- [Spring Boot Web](https://docs.spring.io/spring-boot/docs/current/reference/html/web.html)
