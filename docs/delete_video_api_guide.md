# API Guide: Delete Video

This document provides details on how to use the API endpoint to delete a video generation job.

## Endpoint

`DELETE /v1/videos/{video_id}`

Deletes a video generation job and its associated files from the server. A job cannot be deleted if it is currently being processed.

---

## Input

### Path Parameters

| Parameter  | Type   | Required | Description                             |
| :--------- | :----- | :------- | :-------------------------------------- |
| `video_id` | string | Yes      | The unique identifier of the video job. |

---

## Output

### Success Response (200 OK)

If the deletion is successful, the server responds with a JSON object containing the metadata of the deleted job. The `status` field will be set to `"deleted"`.

**Example Response:**

```json
{
  "id": "video_1721105333_1234",
  "model": "InfinteTalk",
  "status": "deleted",
  "progress": 0,
  "created_at": 1721105333,
  "seconds": "15",
  "duration": 14,
  "estimated_time": 0,
  "queue_length": 0,
  "error": ""
}
```

---

## Error Responses

If the request fails, the server will return a JSON object with an error message and a corresponding HTTP status code.

### 400 Bad Request

This error occurs if you try to delete a job that is currently in the `processing` state.

**Example:**

```json
{
  "error": {
    "message": "Video with id video_1721105333_5678 is processing and cannot be deleted.",
    "code": "400"
  }
}
```

### 404 Not Found

This error occurs if the specified `video_id` cannot be found.

**Reason 1: Video ID does not exist.**

```json
{
  "error": {
    "message": "Video with id video_1721105333_9999 not found.",
    "code": "404"
  }
}
```

**Reason 2: The job queue file is missing on the server.**

```json
{
  "error": {
    "message": "Job queue is missing and video with id video_1721105333_9999 not found.",
    "code": "404"
  }
}
```

### 500 Internal Server Error

This error indicates an unexpected problem on the server side while trying to process the deletion request.

**Example:**

```json
{
  "error": {
    "message": "Internal server error: [details of the error].",
    "code": "500"
  }
}
```
