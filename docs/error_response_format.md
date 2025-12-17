# API 错误响应格式说明

当 API 请求遇到错误时，服务器会返回一个包含特定 HTTP 状态码和 JSON 格式消息的响应。这有助于客户端理解错误的具体原因。

## 通用错误格式

所有错误响应都遵循以下 JSON 结构：

```json
{
  "error": {
    "message": "具体的错误信息描述",
    "code": "错误码"
  }
}
```

- `error.message`: 一个字符串，描述了错误的详细信息。
- `error.code`: 一个字符串形式的 HTTP 状态码，例如 "400", "404", "500"。

---

## 错误类型

### 1. 客户端错误 (HTTP 400 - Bad Request)

当客户端发送的请求包含无效参数时，会触发此类错误。

**触发条件:**

- 缺少必要的参数（例如 `input_reference` 或 `audio`）。
- 参数格式不正确（例如 `size` 不是 `widthxheight` 格式）。
- 参数值无效（例如 `seconds` 小于或等于 0）。
- 上传的音频文件损坏或无法解析时长。

**响应示例:**

```json
{
  "error": {
    "message": "Invalid parameter type: The 'seconds' parameter must be greater than 0.",
    "code": "400"
  }
}
```

### 2. 未找到资源 (HTTP 404 - Not Found)

当请求的资源（例如特定 `video_id` 的视频）在服务器上不存在时，会返回此错误。

**触发条件:**

- 在查询视频状态或获取视频内容时，提供了不存在的 `video_id`。

**响应示例:**

```json
{
  "error": {
    "message": "Video with id video_1721105333_1234 not found.",
    "code": "404"
  }
}
```

### 3. 服务器内部错误 (HTTP 500 - Internal Server Error)

当服务器在处理请求时遇到意外的内部问题时，会返回此错误。

**触发条件:**

- 服务组件加载失败。
- 在视频生成过程中发生未被捕获的异常。
- 其他任何意外的服务器端问题。

**响应示例:**

```json
{
  "error": {
    "message": "Internal server error: Component loader is not initialized.",
    "code": "500"
  }
}
```
